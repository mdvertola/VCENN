import os
import sys
import random
import re
import traceback
from tensorflow.keras.optimizers import RMSprop, Adam
from scipy.stats import rankdata
import math
import numpy as np
from tqdm import tqdm
import argparse
random.seed(42)
import threading
import configs
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

from utils import normalize, pad, convert, revert
import models, configs, data_loader

class SearchEngine:
    def __init__(self, args, conf=None):
        self.data_path = args.data_path + args.dataset+'/'
        self.train_params = conf.get('training_params', dict())
        self.data_params = conf.get('data_params',dict())
        self.model_params = conf.get('model_params',dict())

        self._eval_sets = None

        self._code_reprs = None
        self._codebase = None
        self._codebase_chunksize = 2000000

    ##### Model Loading / saving #####
    def save_model(self, model, epoch):
        model_path = f"./output/{model.__class__.__name__}/models/"
        os.makedirs(model_path, exist_ok=True)
        model.save(model_path + f"epo{epoch}_code.h5", model_path + f"epo{epoch}_desc.h5", overwrite=True)

    def load_model(self, model, epoch):
        model_path = f"./output/{model.__class__.__name__}/models/"
        assert os.path.exists(model_path + f"epo{epoch}_code.h5"),f"Weights at epoch {epoch} not found"
        assert os.path.exists(model_path + f"epo{epoch}_desc.h5"),f"Weights at epoch {epoch} not found"
        model.load(model_path + f"epo{epoch}_code.h5", model_path + f"epo{epoch}_desc.h5")


    ##### Training #####
    def train(self, model):
        if self.train_params['reload']>0:
            self.load_model(model, self.train_params['reload'])
        valid_every = self.train_params.get('valid_every', None)
        save_every = self.train_params.get('save_every', None)
        batch_size = self.train_params.get('batch_size', 128)
        nb_epoch = self.train_params.get('nb_epoch', 10)
        split = self.train_params.get('validation_split', 0)

        val_loss = {'loss': 1., 'epoch': 0}
        chunk_size = self.train_params.get('chunk_size', 100000)

        for i in range(self.train_params['reload']+1, nb_epoch):
            print('Epoch %d :: \n' % i, end='')

            logger.debug('loading data chunk..')
            offset = (i-1)*self.train_params.get('chunk_size', 100000)

            git_methnames = data_loader.load_hdf5(self.data_path+self.data_params['train_git_methname'], offset, chunk_size)
            git_apis = data_loader.load_hdf5(self.data_path+self.data_params['train_git_apiseq'], offset, chunk_size)
            git_tokens = data_loader.load_hdf5(self.data_path+self.data_params['train_git_tokens'], offset, chunk_size)
            stack_methnames = data_loader.load_hdf5(self.data_path+self.data_params['train_stack_methname'], offset, chunk_size)
            stack_apis = data_loader.load_hdf5(self.data_path+self.data_params['train_stack_apiseq'], offset, chunk_size)
            stack_tokens = data_loader.load_hdf5(self.data_path+self.data_params['train_stack_tokens'], offset, chunk_size)
            

            logger.debug('padding data..')
            git_methnames = pad(git_methnames, self.data_params['git_methname_len'])
            git_apiseqs = pad(git_apis, self.data_params['git_apiseq_len'])
            git_tokens = pad(git_tokens, self.data_params['git_tokens_len'])
            stack_methnames = pad(stack_methnames, self.data_params['stack_methname_len'])
            stack_apiseqs = pad(stack_apis, self.data_params['stack_apiseq_len'])
            stack_tokens = pad(stack_tokens, self.data_params['stack_tokens_len'])
            
            
            # good_descs = pad(descs,self.data_params['desc_len'])
            # bad_descs=[desc for desc in descs]
            # random.shuffle(bad_descs)
            # bad_descs = pad(bad_descs, self.data_params['desc_len'])

            hist = model.fit([git_methnames, git_apiseqs, git_tokens, stack_methnames, stack_apiseqs, stack_tokens], epochs=1, batch_size=batch_size, validation_split=split)

            if hist.history['val_loss'][0] < val_loss['loss']:
                val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
            print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'], val_loss['epoch']))

            if save_every is not None and i % save_every == 0:
                self.save_model(model, i)

            if valid_every is not None and i % valid_every == 0:
                acc, mrr, map, ndcg = self.valid(model, 1000, 1)

    ##### Evaluation in the develop set #####
    def valid(self, model, poolsize, K):
        """
        validate in a code pool.
        param: poolsize - size of the code pool, if -1, load the whole test set
        """
        def ACC(real,predict):
            sum=0.0
            for val in real:
                try: index=predict.index(val)
                except ValueError: index=-1
                if index!=-1: sum=sum+1
            return sum/float(len(real))
        def MAP(real,predict):
            sum=0.0
            for id,val in enumerate(real):
                try: index=predict.index(val)
                except ValueError: index=-1
                if index!=-1: sum=sum+(id+1)/float(index+1)
            return sum/float(len(real))
        def MRR(real,predict):
            sum=0.0
            for val in real:
                try: index=predict.index(val)
                except ValueError: index=-1
                if index!=-1: sum=sum+1.0/float(index+1)
            return sum/float(len(real))
        def NDCG(real,predict):
            dcg=0.0
            idcg=IDCG(len(real))
            for i,predictItem in enumerate(predict):
                if predictItem in real:
                    itemRelevance=1
                    rank = i+1
                    dcg+=(math.pow(2,itemRelevance)-1.0)*(math.log(2)/math.log(rank+1))
            return dcg/float(idcg)
        def IDCG(n):
            idcg=0
            itemRelevance=1
            for i in range(n):
                idcg+=(math.pow(2, itemRelevance)-1.0)*(math.log(2)/math.log(i+2))
            return idcg

        #load valid dataset
        if self._eval_sets is None:
            git_methnames = data_loader.load_hdf5(self.data_path+self.data_params['git_methname'], 0, poolsize)
            git_apiseqs= data_loader.load_hdf5(self.data_path+self.data_params['git_apiseq'], 0, poolsize)
            git_tokens = data_loader.load_hdf5(self.data_path+self.data_params['git_tokens'], 0, poolsize)
            stack_methnames = data_loader.load_hdf5(self.data_path+self.data_params['stack_methname'], 0, poolsize)
            stack_apiseqs= data_loader.load_hdf5(self.data_path+self.data_params['stack_apiseq'], 0, poolsize)
            stack_tokens = data_loader.load_hdf5(self.data_path+self.data_params['stack_tokens'], 0, poolsize)
            self._eval_sets={'git_methnames': git_methnames, 'git_apiseqs': git_apiseqs, 'git_tokens': git_tokens, 'stack_methnames': stack_methnames, 'stack_apiseqs': stack_apiseqs, 'stack_tokens':stack_tokens}

        accs,mrrs,maps,ndcgs = [], [], [], []
        data_len = len(self._eval_sets['git_methnames'])
        for i in tqdm(range(data_len)):
            # desc=self._eval_sets['descs'][i]#good desc
            # descs = pad([desc]*data_len,self.data_params['desc_len'])
            git_methnames = pad(self._eval_sets['git_methnames'],self.data_params['git_methname_len'])
            git_apiseqs= pad(self._eval_sets['git_apiseqs'],self.data_params['git_apiseq_len'])
            git_tokens= pad(self._eval_sets['git_tokens'],self.data_params['git_tokens_len'])
            stack_methnames = pad(self._eval_sets['stack_methnames'],self.data_params['stack_methname_len'])
            stack_apiseqs= pad(self._eval_sets['stack_apiseqs'],self.data_params['stack_apiseq_len'])
            stack_tokens= pad(self._eval_sets['stack_tokens'],self.data_params['stack_tokens_len'])
            n_results = K
            sims = model.predict([git_methnames, git_apiseqs, git_tokens,stack_methnames,stack_apiseqs,stack_tokens], batch_size=data_len).flatten()
            negsims= np.negative(sims)
            predict = np.argpartition(negsims, kth=n_results-1)
            predict = predict[:n_results]
            predict = [int(k) for k in predict]
            real=[i]
            accs.append(ACC(real,predict))
            mrrs.append(MRR(real,predict))
            maps.append(MAP(real,predict))
            ndcgs.append(NDCG(real,predict))
        logger.info(f'ACC={np.mean(accs)}, MRR={np.mean(mrrs)}, MAP={np.mean(maps)}, nDCG={np.mean(ndcgs)}')
        return accs,mrrs,maps,ndcgs


    ##### Compute Code Representations for StackOverflow #####
    def repr_code(self, model):
        logger.info('Loading the use data ..')
        
        stack_methnames = data_loader.load_hdf5(self.data_path+self.data_params['use_stack_methname'],0,-1)
        stack_apiseqs = data_loader.load_hdf5(self.data_path+self.data_params['use_stack_apiseq'],0,-1)
        stack_tokens = data_loader.load_hdf5(self.data_path+self.data_params['use_stack_tokens'],0,-1)
        
        stack_methnames = pad(stack_methnames, self.data_params['stack_methname_len'])
        stack_apiseqs = pad(stack_apiseqs, self.data_params['stack_apiseq_len'])
        stack_tokens = pad(stack_tokens, self.data_params['stack_tokens_len'])

        logger.info('Representing code ..')
        
        vecs= model.stack_repr_code([stack_methnames, stack_apiseqs, stack_tokens], batch_size=10000)
        # 
        vecs= vecs.astype(np.float)
        vecs= normalize(vecs)
       
        return vecs

    ##### Github 'search terms' and realtime vector representation #####
    def search(self, model,methnameTokens, apiseqTokens, tokenTokens, git_methnames, git_apiseqs, git_tokens, n_results):
        # 
        
        methnames = np.asarray([convert(methnameTokens, git_methnames)])
        apiseqs = np.asarray([convert(apiseqTokens, git_apiseqs)])
        tokens = np.asarray([convert(tokenTokens, git_tokens)])
        
        padded_methnames = pad(methnames, self.data_params['git_methname_len'])
        padded_apiseqs = pad(apiseqs, self.data_params['git_apiseq_len'])
        padded_tokens = pad(tokens, self.data_params['git_tokens_len'])
        
        
        
        
        
        print('functionTokens: '+ str(methnames))
        print('operationTokens: '+ str(apiseqs))
        print('tokenTokens: '+str(tokens))
        print('__________________________________________________________________________________________________________________')
        logger.info('Representing code ..')
        


        vecs= model.git_repr_code([padded_methnames,padded_apiseqs,padded_tokens], batch_size=10000)
        vecs= vecs.astype(np.float)
        vecs= normalize(vecs)
        git_code_repr = vecs.T

        # print('git_code_repr')
        # print(git_code_repr)
        
        
        codes, sims = [], []
        threads=[]
        for i,code_reprs_chunk in enumerate(self._code_reprs):
            t = threading.Thread(target=self.search_thread, args = (codes,sims,git_code_repr,code_reprs_chunk,i,n_results))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:#wait until all sub-threads finish
            t.join()
        return codes,sims
    
    # def search(self, model, vocab, query, n_results=10):
    #     desc=[convert(vocab, query)]#convert desc sentence to word indices
    #     padded_desc = pad(desc, self.data_params['desc_len'])
    #     desc_repr=model.repr_desc([padded_desc])
    #     desc_repr=desc_repr.astype(np.float32)
    #     desc_repr = normalize(desc_repr).T # [dim x 1]
    #     codes, sims = [], []
    #     threads=[]
    #     for i,code_reprs_chunk in enumerate(self._code_reprs):
    #         t = threading.Thread(target=self.search_thread, args = (codes,sims,desc_repr,code_reprs_chunk,i,n_results))
    #         threads.append(t)
    #     for t in threads:
    #         t.start()
    #     for t in threads:#wait until all sub-threads finish
    #         t.join()
    #     return codes,sims

    def search_thread(self, codes, sims, git_code_reprs, code_reprs,i, n_results):
    #1. compute similarity
        chunk_sims=np.dot(code_reprs,git_code_reprs) # [pool_size x 1]
        chunk_sims = np.squeeze(chunk_sims)
    #2. choose top results
        negsims=np.negative(chunk_sims)
        maxinds = np.argpartition(negsims, kth=n_results-1)
        maxinds = maxinds[:n_results]
        chunk_codes = [self._codebase[i][k] for k in maxinds]
        chunk_sims = chunk_sims[maxinds]
        codes.extend(chunk_codes)
        sims.extend(chunk_sims)

    def postproc(self,codes_sims):
        codes_, sims_ = zip(*codes_sims)
        codes= [code for code in codes_]
        sims= [sim for sim in sims_]
        final_codes=[]
        final_sims=[]
        n=len(codes_sims)
        for i in range(n):
            is_dup=False
            for j in range(i):
                if codes[i][:80]==codes[j][:80] and abs(sims[i]-sims[j])<0.01:
                    is_dup=True
            if not is_dup:
                final_codes.append(codes[i])
                final_sims.append(sims[i])
        return zip(final_codes,final_sims)


def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument("--data_path", type=str, default='./data/', help="working directory")
    parser.add_argument("--model", type=str, default="JointEmbeddingModel", help="model name")
    parser.add_argument("--dataset", type=str, default="stackGit", help="dataset name")
    parser.add_argument("--mode", choices=["train","eval","repr_code","search"], default='train',
                        help="The mode to run. The `train` mode trains a model;"
                        " the `eval` mode evaluat models in a test set "
                        " The `repr_code/repr_desc` mode computes vectors"
                        " for a code snippet or a natural language description with a trained model.")
    parser.add_argument("--verbose",action="store_true", default=True, help="Be verbose")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config=getattr(configs, 'config_'+args.model)()
    engine = SearchEngine(args, config)

    ##### Define model ######
    logger.info('Build Model')
    model = getattr(models, args.model)(config)#initialize the model
    model.build()
    model.summary(export_path = f"./output/{args.model}/")

    optimizer = config.get('training_params', dict()).get('optimizer', 'adam')
    model.compile(optimizer=optimizer)

    data_path = args.data_path+args.dataset+'/'

    if args.mode=='train':
        engine.train(model)

    elif args.mode=='eval': # evaluate for a specific epoch
        if config['training_params']['reload']>0:
            engine.load_model(model, config['training_params']['reload'])
        engine.eval(model, -1, 10)

    elif args.mode=='repr_code':
        if config['training_params']['reload']>0:
            engine.load_model(model, config['training_params']['reload'])
        vecs = engine.repr_code(model)
        data_loader.save_code_reprs(vecs, data_path+config['data_params']['use_codevecs'])

    elif args.mode=='search':
        #search code based on a desc
        if config['training_params']['reload']>0:
            engine.load_model(model, config['training_params']['reload'])
        engine._code_reprs = data_loader.load_code_reprs(data_path+config['data_params']['use_codevecs'], engine._codebase_chunksize)
        engine._codebase = data_loader.load_codebase(data_path+config['data_params']['use_codebase'], engine._codebase_chunksize)
        
        while True:
            try:
                # take git code snippet as input
                print('\n')
                print('Vulnerable Code Clone Detection')
                print('__________________________________________________________________________________________________________________')
               
                print('Find Vulnerable GitHub Code In StackOverflow')
                print('The search expects the following input format: ')
                print('Vulnerable Code Snippet @@@ Relevent Information')
                print('__________________________________________________________________________________________________________________')
                gitInfo = input('Input: ')
                gitInfo = gitInfo.lower()
                cleanGitSnip = gitInfo.split('@@@')[0]
                methnameTokens = data_loader.load_pickle(data_path+config['data_params']['vocab_methname'])
                apiseqTokens = data_loader.load_pickle(data_path+config['data_params']['vocab_apiseq'])
                tokenTokens = data_loader.load_pickle(data_path+config['data_params']['vocab_tokens'])
                
                # Regex Rules to seperate inputted snippets into functions operations and tokens (meth,apiseq,token)
                functionPattern = '[a-zA-Z._]+\(.*?\)+'
                functionNamePattern = '[a-zA-Z._]+\('
                operationsPattern = '\(.*?\)+'
                cleanOperationsPattern = "[^a-zA-Z0-9 .:&%*\-_+=/%><!\[\]]+"
                lettersPattern = "[a-zA-Z]+"

                # seperate snippets and assign proper named variables for later
                gitFunctions = re.findall(functionPattern, cleanGitSnip)
                gitFunctionNames = str(re.findall(functionNamePattern, cleanGitSnip))
                gitFunctionNames = re.findall(lettersPattern, str(gitFunctionNames))
                print('functions: '+ str(gitFunctionNames))

                gitOperations = re.findall(operationsPattern, str(cleanGitSnip))
                gitOperations = re.sub(cleanOperationsPattern,'',str(gitOperations))
                gitOperations = re.findall(lettersPattern,str(gitOperations))
                print('operations: '+ str(gitOperations))
                try:
                    gitTokens = gitInfo.split('@@@')[1]
                    gitTokens = re.findall(lettersPattern,str(gitTokens))
                    print('tokens: '+ str(gitTokens))
                except Exception:
                    print('please input the following format: ')
                    print('vulnerable code snippet @@@ relevant keywords (python, pickle, etc.)')

                # rename to match current model variable names (to be removed later)
                git_methname = gitFunctionNames
                git_apiseq = gitOperations
                git_tokens = gitTokens


                # git_stuff = git_methname + git_apiseq + git_tokens 
                # n_results = int(input('How many results? '))
                n_results=10
            except Exception:
                print("Exception while parsing your input:") 
                traceback.print_exc()
                break
            # query = query.lower().replace('how to ', '').replace('how do i ', '').replace('how can i ', '').replace('?', '').strip()
            codes,sims=engine.search(model, methnameTokens, apiseqTokens, tokenTokens, git_methname, git_apiseq, git_tokens, n_results)
            zipped=zip(codes,sims)
            zipped=sorted(zipped, reverse=True, key=lambda x:x[1])
            zipped=engine.postproc(zipped)
            zipped = list(zipped)[:n_results]
            results = '\n\n'.join(map(str,zipped)) #combine the result into a returning string
            print(results)


    # elif args.mode=='search':
    #     #search code based on a desc
    #     if config['training_params']['reload']>0:
    #         engine.load_model(model, config['training_params']['reload'])
    #     engine._code_reprs = data_loader.load_code_reprs(data_path+config['data_params']['use_codevecs'], engine._codebase_chunksize)
    #     engine._codebase = data_loader.load_codebase(data_path+config['data_params']['use_codebase'], engine._codebase_chunksize)
    #     vocab = data_loader.load_pickle(data_path+config['data_params']['vocab_desc'])
    #     while True:
    #         try:
    #             query = input('Input Vulnerable GitHub Code Snippet: ')
    #             n_results = int(input('How many results? '))
    #         except Exception:
    #             print("Exception while parsing your input:")
    #             traceback.print_exc()
    #             break
    #         query = query.lower().replace('how to ', '').replace('how do i ', '').replace('how can i ', '').replace('?', '').strip()
    #         codes,sims=engine.search(model, vocab, query, n_results)
    #         zipped=zip(codes,sims)
    #         zipped=sorted(zipped, reverse=True, key=lambda x:x[1])
    #         zipped=engine.postproc(zipped)
    #         zipped = list(zipped)[:n_results]
    #         results = '\n\n'.join(map(str,zipped)) #combine the result into a returning string
    #         print(results)