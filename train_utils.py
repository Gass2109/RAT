import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
#import matplotlib.pyplot as plt
#import seaborn
#seaborn.set_context(context="talk")
from architecture import *

device = 'cpu'
import os

class NoamOpt:
    "Optim wrapper that implements rate."
    #512, 1, 400
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if self.warmup == 0:
            return self.factor
        else:
            return self.factor * \
                (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

class Batch_Loss(nn.Module):
    def __init__(self, commission_ratio, interest_rate, gamma=0.1,beta=0.1, size_average=True):
        super(Batch_Loss, self).__init__()
        self.gamma = gamma  #variance penalty
        self.beta = beta
        self.size_average = size_average
        self.commission_ratio=commission_ratio
        self.interest_rate=interest_rate


    def forward(self, w, y):            # w:[128,1,12]   y:[128,11,4] 
        close_price=y[:,:,0:1].to(device)   #   [128,11,1]
        #future close prise (including cash)
        close_price=torch.cat([torch.ones(close_price.size()[0],1,1).to(device),close_price],1).to(device)         #[128,11,1]cat[128,1,1]->[128,12,1]
        reward=torch.matmul(w,close_price)                                                                 #[128,1,1]
        close_price=close_price.view(close_price.size()[0],close_price.size()[2],close_price.size()[1])    #[128,1,12] 
###############################################################################################################
        element_reward=w*close_price
        interest=torch.zeros(element_reward.size(),dtype=torch.float).to(device)
        interest[element_reward<0]=element_reward[element_reward<0]
        interest=torch.sum(interest,2).unsqueeze(2)*self.interest_rate  #[128,1,1]
###############################################################################################################
        future_omega=w*close_price/reward  #[128,1,12]           
        wt=future_omega[:-1]               #[128,1,12]
        wt1=w[1:]                          #[128,1,12]
        pure_pc=1-torch.sum(torch.abs(wt-wt1),-1)*self.commission_ratio   #[128,1]
        pure_pc=pure_pc.to(device)
        pure_pc=torch.cat([torch.ones([1,1]).to(device),pure_pc],0)
        pure_pc=pure_pc.view(pure_pc.size()[0],1,pure_pc.size()[1])       #[128,1,1]
        
        cost_penalty = torch.sum(torch.abs(wt-wt1),-1)
################## Deduct transaction fee ##################
        reward=reward*pure_pc    #reward=pv_vector
################## Deduct loan interest ####################
        reward=reward+interest
        portfolio_value=torch.prod(reward,0)
        batch_loss=-torch.log(reward)
#####################variance_penalty##############################
#        variance_penalty = ((torch.log(reward)-torch.log(reward).mean())**2).mean()
        if self.size_average:
            loss = batch_loss.mean() #+ self.gamma*variance_penalty + self.beta*cost_penalty.mean() 
            return loss, portfolio_value[0][0]
        else:
            loss = batch_loss.mean() #+self.gamma*variance_penalty + self.beta*cost_penalty.mean() #(dim=0)                           
            return loss, portfolio_value[0][0]

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self,  criterion, opt=None):
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y):
        loss, portfolio_value= self.criterion(x,y)         
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss, portfolio_value



def max_drawdown(pc_array):
    """calculate the max drawdown with the portfolio changes
    @:param pc_array: all the portfolio changes during a trading process
    @:return: max drawdown
    """
    portfolio_values = []
    drawdown_list = []
    max_benefit = 0
    for i in range(pc_array.shape[0]):
        if i > 0:
            portfolio_values.append(portfolio_values[i - 1] * pc_array[i])
        else:
            portfolio_values.append(pc_array[i])
        if portfolio_values[i] > max_benefit:
            max_benefit = portfolio_values[i]
            drawdown_list.append(0.0)
        else:
            drawdown_list.append(1.0 - portfolio_values[i] / max_benefit)
    return max(drawdown_list)

class Test_Loss(nn.Module):
    def __init__(self, commission_ratio,interest_rate,gamma=0.1,beta=0.1, size_average=True):
        super(Test_Loss, self).__init__()
        self.gamma = gamma  #variance penalty
        self.beta = beta
        self.size_average = size_average
        self.commission_ratio=commission_ratio
        self.interest_rate=interest_rate

    def forward(self, w, y):               # w:[128,10,1,12] y(128,10,11,4)
        close_price = y[:,:,:,0:1].to(device)    #   [128,10,11,1]
        close_price = torch.cat([torch.ones(close_price.size()[0],close_price.size()[1],1,1).to(device),close_price],2).to(device)       #[128,10,11,1]cat[128,10,1,1]->[128,10,12,1]
        reward = torch.matmul(w,close_price)   #  [128,10,1,12] * [128,10,12,1] ->[128,10,1,1]
        close_price = close_price.view(close_price.size()[0],close_price.size()[1],close_price.size()[3],close_price.size()[2])  #[128,10,12,1] -> [128,10,1,12]
##############################################################################
        element_reward = w*close_price
        interest = torch.zeros(element_reward.size(),dtype = torch.float).to(device)
        interest[element_reward<0] = element_reward[element_reward<0]
#        print("interest:",interest.size(),interest,'\r\n')
        interest = torch.sum(interest,3).unsqueeze(3)*self.interest_rate  #[128,10,1,1]
##############################################################################
        future_omega = w*close_price/reward    #[128,10,1,12]*[128,10,1,12]/[128,10,1,1]
        wt=future_omega[:,:-1]                 #[128, 9,1,12]   
        wt1=w[:,1:]                            #[128, 9,1,12]
        pure_pc=1-torch.sum(torch.abs(wt-wt1),-1)*self.commission_ratio     #[128,9,1]
        pure_pc=pure_pc.to(device)
        pure_pc=torch.cat([torch.ones([pure_pc.size()[0],1,1]).to(device),pure_pc],1)      #[128,1,1] cat  [128,9,1] ->[128,10,1]        
        pure_pc=pure_pc.view(pure_pc.size()[0],pure_pc.size()[1],1,pure_pc.size()[2])  #[128,10,1] ->[128,10,1,1]          
        cost_penalty = torch.sum(torch.abs(wt-wt1),-1)                                 #[128, 9, 1]      
################## Deduct transaction fee ##################
        reward = reward*pure_pc                                                        #[128,10,1,1]*[128,10,1,1]  test: [1,2808-31,1,1]
################## Deduct loan interest ####################
        reward= reward+interest
        if not self.size_average:
            tst_pc_array=reward.squeeze()
            sr_reward=tst_pc_array-1
            SR=sr_reward.mean()/sr_reward.std()
#            print("SR:",SR.size(),"reward.mean():",reward.mean(),"reward.std():",reward.std())
            SN=torch.prod(reward,1) #[1,1,1,1]
            SN=SN.squeeze() #
#            print("SN:",SN.size())
            St_v=[]
            St=1.            
            MDD=max_drawdown(tst_pc_array)
            #print("\n\n\n\n")
            #print(reward.size()[1])
            for k in range(reward.size()[1]):  #2808-31
                St*=reward[0,k,0,0]
                St_v.append(St.item())
                #print(f"at {k}, we have {St.item()}")
            CR=SN/MDD            
            TO=cost_penalty.mean()
##############################################
        portfolio_value=torch.prod(reward,1)     #[128,1,1]
        batch_loss=-torch.log(portfolio_value)   #[128,1,1]

        if self.size_average:
            loss = batch_loss.mean() 
            return loss, portfolio_value.mean()
        else:
            loss = batch_loss.mean() 
            return loss, portfolio_value[0][0][0],SR,CR,St_v,tst_pc_array,TO


class SimpleLossCompute_tst:
    "A simple loss compute and train function."
    def __init__(self,  criterion, opt=None):
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y):
        if self.opt is not None:
            loss, portfolio_value= self.criterion(x,y)
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
            return loss, portfolio_value
        else:
            loss, portfolio_value,SR,CR,St_v,tst_pc_array,TO = self.criterion(x,y)     
            return loss, portfolio_value,SR,CR,St_v,tst_pc_array,TO   



def make_std_mask(local_price_context,batch_size):
    "Create a mask to hide padding and future words."
    local_price_mask = (torch.ones(batch_size,1,1)==1)            
    local_price_mask = local_price_mask & (subsequent_mask(local_price_context.size(-2)).type_as(local_price_mask.data))   
    return local_price_mask    



def train_one_step(DM,x_window_size,model,loss_compute,local_context_length):
    batch=DM.next_batch()
    batch_input = batch["X"]        #(128, 4, 11, 31)
    batch_y = batch["y"]            #(128, 4, 11)
    batch_last_w = batch["last_w"]  #(128, 11)
    batch_w = batch["setw"]     
#############################################################################
    previous_w=torch.tensor(batch_last_w,dtype=torch.float).to(device)
    previous_w=torch.unsqueeze(previous_w,1)                         #[128, 11] -> [128,1,11]
    batch_input=batch_input.transpose((1,0,2,3))
    batch_input=batch_input.transpose((0,1,3,2))
    src=torch.tensor(batch_input,dtype=torch.float).to(device)   
    price_series_mask = (torch.ones(src.size()[1],1,x_window_size)==1)   #[128, 1, 31] 
    currt_price=src.permute((3,1,2,0))                                   #[4,128,31,11]->[11,128,31,4]
    if(local_context_length>1):
        padding_price=currt_price[:,:,-(local_context_length)*2+1:-1,:] 
    else:
        padding_price=None
    currt_price=currt_price[:,:,-1:,:]                                    #[11,128,31,4]->[11,128,1,4]
    trg_mask = make_std_mask(currt_price,src.size()[1])
    batch_y=batch_y.transpose((0,2,1))                                    #[128, 4, 11] ->#[128,11,4]
    trg_y=torch.tensor(batch_y,dtype=torch.float).to(device)
    out = model.forward(src, currt_price, previous_w,  
                        price_series_mask, trg_mask, padding_price)
    new_w=out[:,:,1:]  #去掉cash
    new_w=new_w[:,0,:]  # #[109,1,11]->#[109,11]
    new_w=new_w.detach().cpu().numpy()
    batch_w(new_w)  
    
    loss, portfolio_value = loss_compute(out,trg_y)           
    return loss, portfolio_value


def test_online(DM,x_window_size,model,evaluate_loss_compute,local_context_length):
    tst_batch=DM.get_test_set_online(DM._test_ind[0], DM._test_ind[-1], x_window_size)
    tst_batch_input = tst_batch["X"]         
    tst_batch_y = tst_batch["y"]              
    tst_batch_last_w = tst_batch["last_w"]
    tst_batch_w = tst_batch["setw"]

    tst_previous_w=torch.tensor(tst_batch_last_w,dtype=torch.float).to(device)
    tst_previous_w=torch.unsqueeze(tst_previous_w,1)  

    tst_batch_input=tst_batch_input.transpose((1,0,2,3))
    tst_batch_input=tst_batch_input.transpose((0,1,3,2))

    long_term_tst_src=torch.tensor(tst_batch_input,dtype=torch.float).to(device)      
#########################################################################################
    tst_src_mask = (torch.ones(long_term_tst_src.size()[1],1,x_window_size)==1)   


    long_term_tst_currt_price=long_term_tst_src.permute((3,1,2,0)) 
    long_term_tst_currt_price=long_term_tst_currt_price[:,:,x_window_size-1:,:]   
###############################################################################################    
    tst_trg_mask = make_std_mask(long_term_tst_currt_price[:,:,0:1,:],long_term_tst_src.size()[1])
   

    tst_batch_y=tst_batch_y.transpose((0,3,2,1))  
    tst_trg_y=torch.tensor(tst_batch_y,dtype=torch.float).to(device)
    tst_long_term_w=[]  
    tst_y_window_size=len(DM._test_ind)-x_window_size-1-1
    for j in range(tst_y_window_size+1): #0-9
        tst_src=long_term_tst_src[:,:,j:j+x_window_size,:]
        tst_currt_price=long_term_tst_currt_price[:,:,j:j+1,:]
        if(local_context_length>1):
            padding_price=long_term_tst_src[:,:,j+x_window_size-1-local_context_length*2+2:j+x_window_size-1,:]
            padding_price=padding_price.permute((3,1,2,0))  #[4, 1, 2, 11] ->[11,1,2,4]
        else:
            padding_price=None
        out = model.forward(tst_src, tst_currt_price, tst_previous_w,  #[109,1,11]   [109, 11, 31, 3]) torch.Size([109, 11, 3]
                        tst_src_mask, tst_trg_mask, padding_price)
        if(j==0):
            tst_long_term_w=out.unsqueeze(0)  #[1,109,1,12] 
        else:
            tst_long_term_w=torch.cat([tst_long_term_w,out.unsqueeze(0)],0)
        out=out[:,:,1:]  #去掉cash #[109,1,11]
        tst_previous_w=out
    tst_long_term_w=tst_long_term_w.permute(1,0,2,3) ##[10,128,1,12]->#[128,10,1,12]
    tst_loss, tst_portfolio_value, SR, CR, St_v,tst_pc_array,TO =evaluate_loss_compute(tst_long_term_w,tst_trg_y)  
    return tst_loss, tst_portfolio_value, SR, CR, St_v,tst_pc_array,TO, tst_long_term_w




def test_net(DM, total_step, output_step, x_window_size, local_context_length, model, loss_compute, evaluate_loss_compute, is_trn=True, evaluate=True):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    ####每个epoch开始时previous_w=0
    max_tst_portfolio_value=0

    for i in range(total_step):        
        if(is_trn):
            loss, portfolio_value = train_one_step(DM, x_window_size, model, loss_compute, local_context_length)
            total_loss += loss.item()
        if (i % output_step == 0 and is_trn):  
            elapsed = time.time() - start
            print("Epoch Step: %d| Loss per batch: %f| Portfolio_Value: %f | batch per Sec: %f \r\n" %
                    (i,loss.item(), portfolio_value.item() , output_step / elapsed))
            start = time.time()
#########################################################tst########################################################   
        tst_total_loss=0
        with torch.no_grad():
            if(i % output_step == 0 and evaluate):
                model.eval()
                tst_loss, tst_portfolio_value, SR, CR, St_v,tst_pc_array,TO,tst_long_term_w = test_online(DM,x_window_size, model, evaluate_loss_compute, local_context_length)
                tst_total_loss += tst_loss.item()                                         
                elapsed = time.time() - start
                print("Test: %d Loss: %f| Portfolio_Value: %f | SR: %f | CR: %f | TO: %f |testset per Sec: %f" %
                        (i, tst_loss.item(), tst_portfolio_value.item() ,SR.item(), CR.item(), TO.item(), 1/elapsed))
                start = time.time()
#                portfolio_value_list.append(portfolio_value.item())
        
                if(tst_portfolio_value>max_tst_portfolio_value):
                    max_tst_portfolio_value=tst_portfolio_value
                    log_SR=SR
                    log_CR=CR
                    log_St_v=St_v
                    log_tst_pc_array=tst_pc_array
                #print(log_St_v)
    return max_tst_portfolio_value, log_SR, log_CR, log_St_v, log_tst_pc_array,TO, tst_long_term_w



def test_batch(DM,x_window_size,model,evaluate_loss_compute,local_context_length):
    tst_batch=DM.get_test_set()
    tst_batch_input = tst_batch["X"]       #(128, 4, 11, 31)
    tst_batch_y = tst_batch["y"]
    tst_batch_last_w = tst_batch["last_w"]
    tst_batch_w = tst_batch["setw"]

    tst_previous_w=torch.tensor(tst_batch_last_w,dtype=torch.float).to(device)
    tst_previous_w=torch.unsqueeze(tst_previous_w,1)                    #[2426, 1, 11]
    tst_batch_input=tst_batch_input.transpose((1,0,2,3))
    tst_batch_input=tst_batch_input.transpose((0,1,3,2))
    tst_src=torch.tensor(tst_batch_input,dtype=torch.float).to(device)         
    tst_src_mask = (torch.ones(tst_src.size()[1],1,x_window_size)==1)   #[128, 1, 31]   
    tst_currt_price=tst_src.permute((3,1,2,0))                          #(4,128,31,11)->(11,128,31,3)
#############################################################################
    if(local_context_length>1):
        padding_price=tst_currt_price[:,:,-(local_context_length)*2+1:-1,:]  #(11,128,8,4)
    else:
        padding_price=None
#########################################################################

    tst_currt_price=tst_currt_price[:,:,-1:,:]   #(11,128,31,4)->(11,128,1,4)
    tst_trg_mask = make_std_mask(tst_currt_price,tst_src.size()[1])
    tst_batch_y=tst_batch_y.transpose((0,2,1))   #(128, 4, 11) ->(128,11,4)
    tst_trg_y=torch.tensor(tst_batch_y,dtype=torch.float).to(device)
###########################################################################################################
    tst_out = model.forward(tst_src, tst_currt_price, tst_previous_w, #[128,1,11]   [128, 11, 31, 4]) 
                    tst_src_mask, tst_trg_mask,padding_price)

    tst_loss, tst_portfolio_value=evaluate_loss_compute(tst_out,tst_trg_y) 
    return tst_loss, tst_portfolio_value


def train_net(DM, total_step, output_step, x_window_size, local_context_length, model, model_dir, model_index, loss_compute,evaluate_loss_compute, is_trn=True, evaluate=True):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    ####每个epoch开始时previous_w=0
    max_tst_portfolio_value=0
    for i in range(total_step):
        if(is_trn):
            model.train()
            loss, portfolio_value=train_one_step(DM,x_window_size,model,loss_compute,local_context_length)
            total_loss += loss.item()
        if (i % output_step == 0 and is_trn):  
            elapsed = time.time() - start
            print("Epoch Step: %d| Loss per batch: %f| Portfolio_Value: %f | batch per Sec: %f \r\n" %
                    (i,loss.item(), portfolio_value.item() , output_step / elapsed))
            start = time.time()
#########################################################tst########################################################     
        tst_total_loss=0
        with torch.no_grad():
            if(i % output_step == 0 and evaluate):
                model.eval()
                tst_loss, tst_portfolio_value=test_batch(DM,x_window_size,model,evaluate_loss_compute,local_context_length)
#                tst_loss, tst_portfolio_value=evaluate_loss_compute(tst_out,tst_trg_y)
                tst_total_loss += tst_loss.item()
                elapsed = time.time() - start
                print("Test: %d Loss: %f| Portfolio_Value: %f | testset per Sec: %f \r\n" %
                        (i,tst_loss.item(), tst_portfolio_value.item() , 1/elapsed))
                start = time.time()
                
                if(tst_portfolio_value>max_tst_portfolio_value):
                    max_tst_portfolio_value=tst_portfolio_value
                    torch.save(model, model_dir+'/'+str(model_index)+".pkl")
               #    torch.save(model, model_dir+'/'+str(model_index)+".pkl")
                    print("save model!")
    return tst_loss, tst_portfolio_value
