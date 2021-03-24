'''
        Modified VCENN arch
'''




import os
from tensorflow.keras.layers import Input, Concatenate, Dot, Embedding, Dropout, Lambda, Activation, LSTM, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import numpy as np
import logging
logger = logging.getLogger(__name__)
    
class JointEmbeddingModel:
    def __init__(self, config):
        self.model_params = config.get('model_params', dict())
        self.data_params = config.get('data_params',dict())
        self.stack_methname = Input(shape=(self.data_params['stack_methname_len'],), dtype='int32', name='i_stack_methname')
        self.stack_apiseq= Input(shape=(self.data_params['stack_apiseq_len'],),dtype='int32',name='i_stack_apiseq')
        self.stack_tokens=Input(shape=(self.data_params['stack_tokens_len'],),dtype='int32',name='i_stack_tokens')
        self.git_methname = Input(shape=(self.data_params['git_methname_len'],), dtype='int32', name='i_git_methname')
        self.git_apiseq= Input(shape=(self.data_params['git_apiseq_len'],),dtype='int32',name='i_git_apiseq')
        self.git_tokens=Input(shape=(self.data_params['git_tokens_len'],),dtype='int32',name='i_git_tokens')
        
        # initialize a bunch of variables that will be set later
        self.git_code_repr_model=None
        self.stack_code_repr_model=None        
        self._sim_model = None        
        self._training_model = None
        #self.prediction_model = None       
    
    def build(self):
        '''
        1. Build Git Code Representation Model
        '''
        logger.debug('Building Github Code Representation Model')
        git_methname = Input(shape=(self.data_params['git_methname_len'],), dtype='int32', name='git_methname')
        git_apiseq= Input(shape=(self.data_params['git_apiseq_len'],),dtype='int32',name='git_apiseq')
        git_tokens=Input(shape=(self.data_params['git_tokens_len'],),dtype='int32',name='git_tokens')
        
        ## method name representation ##
        #1.embedding
        git_init_emb_weights = np.load(self.model_params['init_embed_weights_git_methname']) if self.model_params['init_embed_weights_git_methname'] is not None else None
        if git_init_emb_weights is not None: git_init_emb_weights = [git_init_emb_weights]
        git_methname_embedding = Embedding(input_dim=self.data_params['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              weights=git_init_emb_weights,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                              #If True, all subsequent layers in the model must support masking, otherwise an exception will be raised.
                              name='git_methname_embedding')
        git_methname_embedding = git_methname_embedding(git_methname)
        git_methname_embedding_dropout = Dropout(0.25,name='dropout_git_methname_embed')
        git_methname_dropout = git_methname_embedding_dropout(git_methname_embedding)
        #2.rnn
        git_methname_f_rnn = LSTM(self.model_params.get('n_lstm_dims', 128), recurrent_dropout=0.2, 
                     return_sequences=True, name='lstm_git_methname_f')
        
        git_methname_b_rnn = LSTM(self.model_params.get('n_lstm_dims', 128), return_sequences=True, 
                     recurrent_dropout=0.2, name='lstm_git_methname_b',go_backwards=True)        
        git_methname_f_rnn = git_methname_f_rnn(git_methname_dropout)
        git_methname_b_rnn = git_methname_b_rnn(git_methname_dropout)
        dropout = Dropout(0.25,name='dropout_git_methname_rnn')
        git_methname_f_dropout = dropout(git_methname_f_rnn)
        git_methname_b_dropout = dropout(git_methname_b_rnn)
        #3.maxpooling
        git_maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),name='maxpool_git_methname')
        git_methname_pool = Concatenate(name='concat_git_methname_lstms')([git_maxpool(git_methname_f_dropout), git_maxpool(git_methname_b_dropout)])
        git_activation = Activation('tanh',name='active_git_methname')
        git_methname_repr = git_activation(git_methname_pool)
        
        
        ## API Sequence Representation ##
        #1.embedding
        git_apiseq_embedding = Embedding(input_dim=self.data_params['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              #weights=weights,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                                         #If True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_git_apiseq')
        git_apiseq_embedding = git_apiseq_embedding(git_apiseq)
        dropout = Dropout(0.25,name='dropout_apiseq_embed')
        git_apiseq_dropout = dropout(git_apiseq_embedding)
        #2.rnn
        git_apiseq_f_rnn = LSTM(self.model_params.get('n_lstm_dims', 100), return_sequences=True, recurrent_dropout=0.2,
                      name='git_lstm_apiseq_f')
        git_apiseq_b_rnn = LSTM(self.model_params.get('n_lstm_dims', 100), return_sequences=True, recurrent_dropout=0.2, 
                      name='git_lstm_apiseq_b', go_backwards=True)        
        git_apiseq_f_rnn = git_apiseq_f_rnn(git_apiseq_dropout)
        git_apiseq_b_rnn = git_apiseq_b_rnn(git_apiseq_dropout)
        dropout = Dropout(0.25,name='dropout_apiseq_rnn')
        git_apiseq_f_dropout = dropout(git_apiseq_f_rnn)
        git_apiseq_b_dropout = dropout(git_apiseq_b_rnn)
        #3.maxpooling
        git_apiseq_maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),name='maxpool_git_apiseq')
        git_apiseq_pool = Concatenate(name='git_concat_apiseq_lstms')([git_apiseq_maxpool(git_apiseq_f_dropout), git_apiseq_maxpool(git_apiseq_b_dropout)])
        git_apiseq_activation = Activation('tanh',name='active_git_apiseq')
        git_apiseq_repr = git_apiseq_activation(git_apiseq_pool)
        
        
        ## Tokens Representation ##
        #1.embedding
        git_init_emb_weights = np.load(self.model_params['init_embed_weights_git_tokens']) if self.model_params['init_embed_weights_git_tokens'] is not None else None
        if git_init_emb_weights is not None: git_init_emb_weights = [git_init_emb_weights]
        git_tokens_embedding = Embedding(input_dim=self.data_params['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              weights=git_init_emb_weights,
                              #mask_zero=True,#Whether 0 in the input is a special "padding" value that should be masked out. 
                              #If True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_tokens')
        git_tokens_embedding = git_tokens_embedding(git_tokens)
        dropout = Dropout(0.25,name='git_dropout_tokens_embed')
        git_tokens_dropout= dropout(git_tokens_embedding)

        #4.maxpooling
        git_tokens_maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),name='git_maxpool_tokens')
        git_tokens_pool = git_tokens_maxpool(git_tokens_dropout)
        git_tokens_activation = Activation('tanh',name='git_active_tokens')
        git_tokens_repr= git_tokens_activation(git_tokens_pool)        
        
        ## concatenate the representation of code ##
        git_merged_methname_api=Concatenate(name='merge_methname_api')([git_methname_repr,git_apiseq_repr])
        git_merged_code_repr=Concatenate(name='merge_coderepr')([git_merged_methname_api,git_tokens_repr])
        git_code_repr=Dense(self.model_params.get('n_hidden',400),activation='tanh',name='git_dense_coderepr')(git_merged_code_repr)
        
        
        self.git_code_repr_model=Model(inputs=[git_methname,git_apiseq,git_tokens],outputs=[git_code_repr],name='git_code_repr_model')     
        
        
        '''
        1. Build Stack Code Representation Model
        '''
        logger.debug('Building Github Code Representation Model')
        stack_methname = Input(shape=(self.data_params['stack_methname_len'],), dtype='int32', name='stack_methname')
        stack_apiseq= Input(shape=(self.data_params['stack_apiseq_len'],),dtype='int32',name='stack_apiseq')
        stack_tokens=Input(shape=(self.data_params['stack_tokens_len'],),dtype='int32',name='stack_tokens')
        
        ## method name representation ##
        #1.embedding
        stack_init_emb_weights = np.load(self.model_params['init_embed_weights_stack_methname']) if self.model_params['init_embed_weights_stack_methname'] is not None else None
        if stack_init_emb_weights is not None: stack_init_emb_weights = [stack_init_emb_weights]
        stack_methname_embedding = Embedding(input_dim=self.data_params['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              weights=stack_init_emb_weights,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                              #If True, all subsequent layers in the model must support masking, otherwise an exception will be raised.
                              name='embedding_stack_methname')
        stack_methname_embedding = stack_methname_embedding(stack_methname)
        dropout = Dropout(0.25,name='dropout_stack_methname_embed')
        stack_methname_dropout = dropout(stack_methname_embedding)
        #2.rnn
        stack_methname_f_rnn = LSTM(self.model_params.get('n_lstm_dims', 128), recurrent_dropout=0.2, 
                     return_sequences=True, name='lstm_stack_methname_f')
        
        stack_methname_b_rnn = LSTM(self.model_params.get('n_lstm_dims', 128), return_sequences=True, 
                     recurrent_dropout=0.2, name='lstm_stack_methname_b',go_backwards=True)        
        stack_methname_f_rnn = stack_methname_f_rnn(stack_methname_dropout)
        stack_methname_b_rnn = stack_methname_b_rnn(stack_methname_dropout)
        stack_dropout = Dropout(0.25,name='dropout_stack_methname_rnn')
        stack_methname_f_dropout = stack_dropout(stack_methname_f_rnn)
        stack_methname_b_dropout = stack_dropout(stack_methname_b_rnn)
        #3.maxpooling
        stack_methname_maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),name='maxpool_stack_methname')
        stack_methname_pool = Concatenate(name='concat_stack_methname_lstms')([stack_methname_maxpool(stack_methname_f_dropout), stack_methname_maxpool(stack_methname_b_dropout)])
        stack_methname_activation = Activation('tanh',name='active_stack_methname')
        stack_methname_repr = stack_methname_activation(stack_methname_pool)
        
        
        ## API Sequence Representation ##
        #1.embedding
        embedding = Embedding(input_dim=self.data_params['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              #weights=weights,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                                         #If True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_stack_apiseq')
        stack_apiseq_embedding = embedding(stack_apiseq)
        stack_apiseq_dropout = Dropout(0.25,name='dropout_apiseq_embed')
        stack_apiseq_dropout = stack_apiseq_dropout(stack_apiseq_embedding)
        #2.rnn
        stack_apiseq_f_rnn = LSTM(self.model_params.get('n_lstm_dims', 100), return_sequences=True, recurrent_dropout=0.2,
                      name='stack_lstm_apiseq_f')
        stack_apiseq_b_rnn = LSTM(self.model_params.get('n_lstm_dims', 100), return_sequences=True, recurrent_dropout=0.2, 
                      name='stack_lstm_apiseq_b', go_backwards=True)        
        stack_apiseq_f_rnn = stack_apiseq_f_rnn(stack_apiseq_dropout)
        stack_apiseq_b_rnn = stack_apiseq_b_rnn(stack_apiseq_dropout)
        stack_dropout = Dropout(0.25,name='dropout_apiseq_rnn')
        stack_apiseq_f_dropout = stack_dropout(stack_apiseq_f_rnn)
        stack_apiseq_b_dropout = stack_dropout(stack_apiseq_b_rnn)
        #3.maxpooling
        stack_apiseq_maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),name='maxpool_stack_apiseq')
        stack_apiseq_pool = Concatenate(name='stack_concat_apiseq_lstms')([stack_apiseq_maxpool(stack_apiseq_f_dropout), stack_apiseq_maxpool(stack_apiseq_b_dropout)])
        activation = Activation('tanh',name='active_stack_apiseq')
        stack_apiseq_repr = activation(stack_apiseq_pool)
        
        
        ## Tokens Representation ##
        #1.embedding
        stack_init_emb_weights = np.load(self.model_params['init_embed_weights_stack_tokens']) if self.model_params['init_embed_weights_stack_tokens'] is not None else None
        if stack_init_emb_weights is not None: stack_init_emb_weights = [stack_init_emb_weights]
        stack_tokens_embedding = Embedding(input_dim=self.data_params['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              weights=stack_init_emb_weights,
                              #mask_zero=True,#Whether 0 in the input is a special "padding" value that should be masked out. 
                              #If True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='stack_tokens_embedding')
        stack_tokens_embedding = stack_tokens_embedding(stack_tokens)
        dropout = Dropout(0.25,name='stack_dropout_tokens_embed')
        stack_tokens_dropout= dropout(stack_tokens_embedding)

        #4.maxpooling
        stack_tokens_maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),name='stack_maxpool_tokens')
        stack_tokens_pool = stack_tokens_maxpool(stack_tokens_dropout)
        stack_tokens_activation = Activation('tanh',name='stack_active_tokens')
        stack_tokens_repr= stack_tokens_activation(stack_tokens_pool)        
        
        ## concatenate the representation of code ##
        stack_merged_methname_api=Concatenate(name='merge_methname_api')([stack_methname_repr,stack_apiseq_repr])
        stack_merged_code_repr=Concatenate(name='merge_coderepr')([stack_merged_methname_api,stack_tokens_repr])
        stack_code_repr=Dense(self.model_params.get('n_hidden',400),activation='tanh',name='stack_dense_coderepr')(stack_merged_code_repr)
        
        
        self.stack_code_repr_model=Model(inputs=[stack_methname,stack_apiseq,stack_tokens],outputs=[stack_code_repr],name='stack_code_repr_model')




        """
        3: calculate the cosine similarity between code and desc
        """     
        logger.debug('Building similarity model') 
        git_code_repr=self.git_code_repr_model([git_methname,git_apiseq,git_tokens])
        stack_code_repr=self.stack_code_repr_model([stack_methname,stack_apiseq,stack_tokens])
        cos_sim=Dot(axes=1, normalize=True, name='cos_sim')([git_code_repr, stack_code_repr])
        
        sim_model = Model(inputs=[git_methname,git_apiseq,git_tokens,stack_methname,stack_apiseq,stack_tokens], outputs=[cos_sim],name='sim_model')   
        self._sim_model=sim_model  #for model evaluation  

        
        '''
        4:Build training model
        '''
        # MATT COMMENT - I removed _good and _bad at the end of what used to be the code desc embeddings with CODEnn
        good_sim = sim_model([self.git_methname,self.git_apiseq,self.git_tokens,self.stack_methname,self.stack_apiseq,self.stack_tokens])# similarity of good output
        bad_sim = sim_model([self.git_methname,self.git_apiseq,self.git_tokens,self.stack_methname,self.stack_apiseq,self.stack_tokens])#similarity of bad output
        loss = Lambda(lambda x: K.maximum(1e-6, self.model_params['margin'] - x[0] + x[1]),
                     output_shape=lambda x: x[0], name='loss')([good_sim, bad_sim])

        logger.debug('Building training model')
        self._training_model=Model(inputs=[self.git_methname,self.git_apiseq,self.git_tokens,self.stack_methname,self.stack_apiseq,self.stack_tokens],
                                   outputs=[loss],name='training_model')
        
                
    def summary(self, export_path):
        print('Summary of the GitHub code representation model')
        self.git_code_repr_model.summary()
        #plot_model(self._code_repr_model, show_shapes=True, to_file= export_path+'code_repr_model.png')  
        print('Summary of the StackOverflow code representation model')
        self.stack_code_repr_model.summary()
        #plot_model(self._desc_repr_model, show_shapes=True, to_file=export_path+'desc_repr_model.png') 
        print ("Summary of the similarity model")
        self._sim_model.summary() 
        #plot_model(self._sim_model, show_shapes=True, to_file= export_path+'sim_model.png')
        print ('Summary of the training model')
        self._training_model.summary()      
        #plot_model(self._training_model, show_shapes=True, to_file=export_path+'training_model.png')  
   

    def compile(self, optimizer, **kwargs):
        logger.info('compiling models')
        self.git_code_repr_model.compile(loss='cosine_similarity', optimizer=optimizer, **kwargs)
        self.stack_code_repr_model.compile(loss='cosine_similarity', optimizer=optimizer, **kwargs)
        self._training_model.compile(loss=lambda y_true, y_pred: y_pred+y_true-y_true, optimizer=optimizer, **kwargs)
        #+y_true-y_true is for avoiding an unused input warning, it can be simply +y_true since y_true is always 0 in the training set.
        self._sim_model.compile(loss='binary_crossentropy', optimizer=optimizer, **kwargs)

    def fit(self, x, **kwargs):
        assert self._training_model is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=x[0].shape[:1],dtype=np.float32)
        return self._training_model.fit(x, y, **kwargs)

    def git_repr_code(self, meth, apiseq, token, **kwargs):
        gitAll = meth + apiseq + token
        return self.git_code_repr_model.predict(gitAll, **kwargs)
    
    def stack_repr_code(self, x, **kwargs):
        return self.stack_code_repr_model.predict(x, **kwargs)
    
    def predict(self, x, **kwargs):
        return self._sim_model.predict(x, **kwargs)
    
    # def git_repr_code(self, x, **kwargs):
    #     return self.git_code_repr_model.predict(x, **kwargs)
    
    # def stack_repr_code(self, x, **kwargs):
    #     return self.stack_code_repr_model.predict(x, **kwargs)
    
    # def predict(self, x, **kwargs):
    #     return self._sim_model.predict(x, **kwargs)

    def save(self, git_code_model_file, stack_code_model_file, **kwargs):
        assert self.git_code_repr_model is not None, 'Must compile the model before saving weights'
        self.git_code_repr_model.save_weights(git_code_model_file, **kwargs)
        assert self.stack_code_repr_model is not None, 'Must compile the model before saving weights'
        self.stack_code_repr_model.save_weights(stack_code_model_file, **kwargs)

    def load(self, git_code_model_file, stack_code_model_file, **kwargs):
        assert self.git_code_repr_model is not None, 'Must compile the model loading weights'
        self.git_code_repr_model.load_weights(git_code_model_file, **kwargs)
        assert self.stack_code_repr_model is not None, 'Must compile the model loading weights'
        self.stack_code_repr_model.load_weights(stack_code_model_file, **kwargs)

 
 
 
 
