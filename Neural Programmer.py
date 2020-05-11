import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
import torch.optim as optim
import nltk
from nltk.corpus import stopwords
import string
import random as rd
import pickle
col_num = 5 #Decide the size of data
huber = 10 #Parameter of loss function
lamb = 25 #Parameter of loss function
alpha = 0.01 #Decide the learning rate
op_words = ['sum','count','max','min','product','print']
cp_words = ['greater','lesser']
lg_words = ['and','or','nand','nor']
sc_words = ['scale']
ew_words = ['ewpower']
ar_words = ['diff']
rs_words = ['reset']
col_ind = ['A','B','C','D','E','F','G','H','I','J']
col_ind = col_ind[:col_num]
all_words = [op_words,cp_words,lg_words,ar_words,sc_words,ew_words,rs_words,col_ind]

def vectorize(words): #Initialize the word embedding 
    dict = {}
    allw = []
    allop = []
    allcol = []
    for wset in words:
        allw.extend(wset)
    len_dict = len(allw)
    for i in range(len_dict):
        vec = np.zeros(len_dict).tolist()
        vec[i] = 1.0
        dict[allw[i]] = vec
        if(allw[i] not in col_ind):
            allop.append(vec)
        else:
            allcol.append(vec)
    return dict, allop, allcol

word_dict, ops, cols = vectorize(all_words)
op_ori = np.array(ops)
col_ori = np.array(cols)
op_mat = Variable(torch.from_numpy(np.array(ops)),requires_grad=True) #Operation Matrix to be learned
col_mat = Variable(torch.from_numpy(np.array(cols)),requires_grad=True) #Column Matrix to be learned

def question_generate_threecol(dataset): #Generate questions of selecting three columns
    row_num, col_num = dataset.shape
    qaset = []
    col_set = []
    for i in range(col_num):
        for j in range(col_num):
            for k in range(col_num):
                if((i != j) and (j != k) and (i != k)):
                    col_set.append((col_ind[i],col_ind[j],col_ind[k]))
    varset = []
    for x in range(10):
        num1 = (10 * np.random.randn(1, 1))[0, 0]
        num2 = (10 * np.random.randn(1, 1))[0, 0]
        varset.append((num1,num2))

    for oneset in col_set:
        col1, col2, col3 = oneset
        colval1 = dataset[:,col_ind.index(col1)].tolist()
        colval2 = dataset[:, col_ind.index(col2)].tolist()
        colval3 = dataset[:, col_ind.index(col3)].tolist()
        for cp1 in cp_words:
            for cp2 in cp_words:
                for lg in lg_words:
                    for op in op_words:
                        for var in varset:
                            var1, var2 = var
                            qone = [cp1,var1,col1,lg,cp2,var2,col2,op,col3]
                            tone = 1
                            if(cp1 == 'greater'):
                                col_selec1 = [elem for elem in colval1 if (elem > var1)]
                            if (cp1 == 'lesser'):
                                col_selec1 = [elem for elem in colval1 if (elem < var1)]
                            if (cp2 == 'greater'):
                                col_selec2 = [elem for elem in colval2 if (elem > var2)]
                            if (cp2 == 'lesser'):
                                col_selec2 = [elem for elem in colval2 if (elem < var2)]
                            ind1 = [colval1.index(p) for p in col_selec1]
                            ind2 = [colval2.index(p) for p in col_selec2]
                            if (lg == 'and'):
                                ind = list(set(ind1)&set(ind2))
                            if (lg == 'or'):
                                ind = list(set(ind1)|set(ind2))
                            if (lg == 'nand'):
                                index = list(set(ind1)&set(ind2))
                                ind = [ig for ig in range(col_num) if ig not in index]
                            if (lg == 'nor'):
                                index = list(set(ind1)|set(ind2))
                                ind = [ig for ig in range(col_num) if ig not in index]
                            col_selec3 = [colval3[x] for x in ind]
                            if (len(col_selec3) == 0):
                                aone = 0
                            else:
                                if (op == 'print'):
                                    tone = 0
                                    aone = np.zeros(dataset.shape)
                                    for r in range(dataset.shape[0]):
                                        if(r in ind):
                                            aone[r,col_ind.index(col3)] = dataset[r,col_ind.index(col3)]
                                if (op == 'sum'):
                                    aone = np.sum(col_selec3)
                                if (op == 'max'):
                                    aone = np.max(col_selec3)
                                if (op == 'min'):
                                    aone = np.min(col_selec3)
                                if (op == 'count'):
                                    aone = len(col_selec3)
                                if (op == 'product'):
                                    aone = np.product(col_selec3)
                            qaset.append((qone,aone,tone))
                            print(qone)
    return qaset

def question_generate_twocol(dataset): #Generating questions of selecting two columns
    row_num, col_num = dataset.shape
    qaset = []
    col_set = []
    for i in range(col_num):
        for j in range(col_num):
            if (i != j):
                col_set.append((col_ind[i], col_ind[j]))
    varset = []
    for x in range(100):
        num1 = (2.5 * np.random.randn(1, 1) + 3)[0, 0]
        num2 = (2.5 * np.random.randn(1, 1) + 3)[0, 0]
        varset.append((num1, num2))

    for oneset in col_set:
        col1, col2 = oneset
        colval1 = dataset[:, col_ind.index(col1)].tolist()
        colval2 = dataset[:, col_ind.index(col2)].tolist()
        for cp1 in cp_words:
            for cp2 in cp_words:
                for lg in lg_words:
                    for ew in ew_words:
                        for var in varset:
                            var1, var2 = var
                            qone = [cp1, var1, col1, lg, cp2, var2, col2, ew]
                            if (cp1 == 'greater'):
                                col_selec1 = [elem for elem in colval1 if (elem > var1)]
                            if (cp1 == 'lesser'):
                                col_selec1 = [elem for elem in colval1 if (elem < var1)]
                            if (cp2 == 'greater'):
                                col_selec2 = [elem for elem in colval2 if (elem > var2)]
                            if (cp2 == 'lesser'):
                                col_selec2 = [elem for elem in colval2 if (elem < var2)]
                            ind1 = [colval1.index(p) for p in col_selec1]
                            ind2 = [colval2.index(p) for p in col_selec2]
                            if (lg == 'and'):
                                ind = list(set(ind1) & set(ind2))
                            if (lg == 'or'):
                                ind = list(set(ind1) | set(ind2))
                            if (lg == 'nand'):
                                index = list(set(ind1) & set(ind2))
                                ind = [ig for ig in range(col_num) if ig not in index]
                            if (lg == 'nor'):
                                index = list(set(ind1) | set(ind2))
                                ind = [ig for ig in range(col_num) if ig not in index]
                            data1 = [colval1[x] for x in ind]
                            data2 = [colval2[y] for y in ind]
                            aone = np.multiply(data1,data2)
                            qaset.append((qone, aone))
                            print(qone, aone)
    return qaset

def question_generate_onecol(dataset): #Generating questions of selecting one column
    row_num, col_num = dataset.shape
    qaset = []
    col_set = []
    for i in range(col_num):
        col_set.append(col_ind[i])

    varset = []
    for x in range(1000):
        num1 = (10 * np.random.randn(1, 1))[0, 0]
        num2 = (10 * np.random.randn(1, 1))[0, 0]
        varset.append((num1,num2))

    for oneset in col_set:
        col1 = oneset
        colval1 = dataset[:, col_ind.index(col1)].tolist()
        for cp1 in cp_words:
            for op in op_words:
                for sc in sc_words:
                    for var in varset:
                        var1, var2 = var
                        qone = [cp1, var1, col1, sc, var2, op]
                        tone = 1
                        if (cp1 == 'greater'):
                            col_selec3 = [var2 * elem for elem in colval1 if (elem > var1)]
                        if (cp1 == 'lesser'):
                            col_selec3 = [var2 * elem for elem in colval1 if (elem < var1)]
                        if (len(col_selec3) == 0):
                            aone = 0
                        else:
                            if (op == 'print'):
                                tone = 0
                                aone = np.zeros(dataset.shape)
                                for r in range(dataset.shape[0]):
                                    if (dataset[r, col_ind.index(col1)] in col_selec3):
                                        aone[r, col_ind.index(col1)] = dataset[r, col_ind.index(col1)]
                            if (op == 'sum'):
                                aone = np.sum(col_selec3)
                            if (op == 'max'):
                                aone = np.max(col_selec3)
                            if (op == 'min'):
                                aone = np.min(col_selec3)
                            if (op == 'count'):
                                aone = len(col_selec3)
                            if (op == 'product'):
                                aone = np.product(col_selec3)
                        qaset.append((qone, aone, tone))
                        print(qone)
    return qaset

def question_generate_arith(dataset): #Generating questions without any other immediate numbers
    row_num, col_num = dataset.shape
    qaset = []
    col_set = []
    for i in range(col_num):
        for j in range(col_num):
            col_set.append((col_ind[i], col_ind[j]))
    for oneset in col_set:
        col1, col2 = oneset
        colval1 = dataset[:, col_ind.index(col1)].tolist()
        colval2 = dataset[:, col_ind.index(col2)].tolist()
        for op1 in op_words:
            for op2 in op_words:
                for ar in ar_words:
                    if(op1 != 'print'):
                        if(op2 != 'print'):
                            qone = [op1, col1, op2, col2, ar]
                            tone = 1
                            if (op1 == 'sum'):
                                data1 = np.sum(colval1)
                            if (op1 == 'max'):
                                data1 = np.max(colval1)
                            if (op1 == 'min'):
                                data1 = np.min(colval1)
                            if (op1 == 'count'):
                                data1 = len(colval1)
                            if (op1 == 'product'):
                                data1 = np.product(colval1)
                            if (op2 == 'sum'):
                                data2 = np.sum(colval2)
                            if (op2 == 'max'):
                                data2 = np.max(colval2)
                            if (op2 == 'min'):
                                data2 = np.min(colval2)
                            if (op2 == 'count'):
                                data2 = len(colval2)
                            if (op2 == 'product'):
                                data2 = np.product(colval2)
                            aone = data1 - data2
                            qaset.append((qone, aone,tone))
                            print(qone)
    return qaset

def question_change(q): #Change question words into embeddings, pick numbers out. 
    op_matnp = op_mat.data.numpy()
    col_matnp = col_mat.data.numpy()
    vws = []
    num = []
    for oneword in q:
        if(oneword in word_dict.keys()):
            emblist = np.array(word_dict[oneword]).reshape((1, len(word_dict[oneword])))
            if(oneword in col_ind):
                for m in range(col_ori.shape[0]):
                    if((col_ori[m,:].tolist() == emblist.tolist()[0])):
                        vws.append(col_matnp[m,:].reshape(1,len(word_dict[oneword])).T)
            else:
                for n in range(op_ori.shape[0]):
                    if((op_ori[n,:].tolist() == emblist.tolist()[0])):
                        vws.append(op_matnp[n,:].reshape(1,len(word_dict[oneword])).T)
        else:
            num.append((vws[-1],oneword))
    return vws, num

def generate_data(M,C): #Get the data columns
    table = 20 * np.random.random_sample((M,C)) - 10
    return table

def question_rnn_forward(qs,wq): #Question RNN module
    #qs:list of (d,1)
    #wq:(d,2d)
    d = qs[0].shape[0]
    zs = []
    z0 = np.zeros(shape=(d,1))
    #q0 = np.concatenate((z0,qs[0]),axis=0)
    for i in range(len(qs)):
        qnow = np.concatenate((z0,qs[i]),axis=0)
        zs.append(z0)
        znow = np.tanh(np.dot(wq.detach().numpy(),qnow))
        z0 = znow
    zs.append(z0)
    q = zs[-1]
    return q, zs

def softmax(inp): #Softmax module
    inp_ex = np.exp(inp)
    inp_sum = np.sum(inp)
    sftm = inp_ex / inp_sum
    return sftm

def selector_op_forward(wop,qr,ht): #Operation selector module
    #qr:(d,1)
    #wop:(d,2d)
    #ht:(d,1), from history RNN
    inp_now = np.concatenate((qr,ht),axis=0)
    alpha_op = softmax(np.dot(op_mat.detach().numpy(),np.tanh(np.dot(wop.detach().numpy(),inp_now))))
    #(15,1)
    return alpha_op

def selector_data_forward(wcol,qr,ht): #Data selector module
    #colmat:(C,d)
    #wcol:(d,2d)
    inp_now = np.concatenate((qr, ht), axis=0)
    alpha_col = softmax(np.dot(col_mat.detach().numpy(),np.tanh(np.dot(wcol.detach().numpy(),inp_now))))
    #(C,1)
    return alpha_col

def operation_forward(dataset,sans,lans,rslc,zs,aop,acol): #Operation Processing module
    #sans[t-2]:const
    #lans[t-2]:(M,C)
    #rslc[t-2]:(M,1)
    #lists of t-1
    #op:(1,d)
    #z:(n,d)
    qn = []
    z = []
    for (x, y) in zs:
        qn.append(y)
        z.append(x)
    try:
        qn_mat = np.array(qn).reshape((len(qn), 1))
        z_mat = np.array(z).reshape((len(z), len(z[0])))
        gop = np.array(word_dict['greater']).reshape((1, len(word_dict['greater'])))
        lop = np.array(word_dict['lesser']).reshape((1, len(word_dict['lesser'])))
        sop = np.array(word_dict['scale']).reshape((1, len(word_dict['scale'])))
        beta_gop = softmax(np.dot(z_mat, gop.T))  # (n,1)
        pivot_gop = np.sum(np.multiply(beta_gop, qn_mat))
        beta_lop = softmax(np.dot(z_mat, lop.T))  # (n,1)
        pivot_lop = np.sum(np.multiply(beta_lop, qn_mat))
        beta_sop = softmax(np.dot(z_mat, sop.T))  # (n,1)
        pivot_sop = np.sum(np.multiply(beta_sop, qn_mat))
    except:
        pivot_gop = 1000
        pivot_lop = -1000
        pivot_sop = 0
    M,C = dataset.shape
    sumt = np.zeros((1,C))
    productt = np.ones((1, C))
    mint = 1000 * np.ones((1, C))
    maxt = (-1000) * np.ones((1, C))
    countt = 0
    try:
        difft = sans[-3] - sans[-1]
    except:
        difft = sans[0] - sans[-1]
    scalet = lans[-1] * pivot_sop
    gt = np.zeros((M,C))
    lt = np.zeros((M,C))
    assn = np.zeros((M,C))
    andt = np.zeros((M,1))
    ort = np.zeros((M,1))
    nant = np.zeros((M, 1))
    nort = np.zeros((M,1))
    rest = np.zeros((M,1))
    for i in range(M):
        try:
            andt[i,0] = min(rslc[-1][i,0],rslc[-2][i,0])
            ort[i, 0] = max(rslc[-1][i, 0], rslc[-2][i, 0])
        except:
            andt[i, 0] = min(rslc[-1][i, 0], rslc[0][i, 0])
            ort[i, 0] = max(rslc[-1][i, 0], rslc[0][i, 0])
        nant[i,0] = abs(andt[i,0] - 1)
        nort[i,0] = abs(ort[i,0] - 1)
        rest[i,0] = 1
        for j in range(C):
            sumt[0,j] += rslc[-1][i,0] * dataset[i,j]
            productt[0, j] = (rslc[-1][i, 0] * dataset[i, j]) * productt[0, j]
            assn[i,j] = rslc[-1][i,0]
            if (dataset[i,j] > maxt[0,j]):
                maxt[0,j] = rslc[-1][i,0] * dataset[i,j]
            if (dataset[i,j] < mint[0,j]):
                mint[0,j] = rslc[-1][i,0] * dataset[i,j]
            countt += rslc[-1][i,0]
            if(dataset[i,j] > pivot_gop):
                gt[i,j] = dataset[i,j]
            if(dataset[i,j] < pivot_lop):
                lt[i,j] = dataset[i,j]
    scalar_ans = aop[word_dict['count'].index(1),0] * countt + aop[word_dict['diff'].index(1),0] * difft
    temp_max = []
    temp_min = []
    temp_pro = []
    for c in range(C):
        scalar_ans += aop[word_dict['sum'].index(1),0] * sumt[0,c] * acol[cols[c].index(1) - len(ops),0]
        temp_max.append(aop[word_dict['max'].index(1),0] * maxt[0,c] * acol[cols[c].index(1) - len(ops),0])
        temp_min.append(aop[word_dict['min'].index(1), 0] * mint[0, c] * acol[cols[c].index(1) - len(ops), 0])
        temp_pro.append(aop[word_dict['product'].index(1), 0] * productt[0, c] * acol[cols[c].index(1) - len(ops), 0])
    #scalar_ans = scalar_ans + max(temp_max) + min(temp_min) + np.product(np.array(temp_pro))
    #lookup_ans = aop[word_dict['scale'].index(1),0] * scalet
    lookup_ans = np.zeros((M,C))
    row_select = np.zeros((M,1))
    for m in range(M):
        row_select[m,0] = aop[word_dict['and'].index(1),0] * andt[m,0] + aop[word_dict['or'].index(1),0] * ort[m,0] + aop[word_dict['nand'].index(1),0] * nant[m,0] + aop[word_dict['nor'].index(1),0] * nort[m,0]
        row_select[m,0] += aop[word_dict['reset'].index(1),0] * rest[m,0]
        for n in range(C):
            lookup_ans[m,n] += aop[word_dict['print'].index(1),0] * acol[cols[n].index(1) - len(ops), 0] * assn[m,n]
            row_select[m,0] += acol[cols[n].index(1) - len(ops), 0] * (aop[word_dict['greater'].index(1),0] * gt[m,n] + aop[word_dict['lesser'].index(1),0] * lt[m,n])
    sans.append(scalar_ans)
    lans.append(lookup_ans)
    rslc.append(row_select)
    return sans, lans, rslc

def history_RNN_forward(aop,acol,whis,htp,t,d): #History RNN module
    #aop:(15,1)
    #acol:(C,1)
    #op_mat:(15,d)
    #col_mat:(C,d)
    #htp:(d,1)
    if(t == 0):
        ct = np.zeros((2*d,1))
    else:
        ophis = np.dot(aop.T,op_mat.detach().numpy()).T #(d,1)
        colhis = np.dot(acol.T,col_mat.detach().numpy()).T #(d,1)
        ct = np.concatenate((ophis,colhis),axis=0)
    inpnow = np.concatenate((ct,htp),axis=0) #(3d,1)
    ht = np.tanh(np.dot(whis.detach().numpy(),inpnow))
    return ht

#what need to be trained: wq, whis, op_mat, col_mat, wcol, wop
def loss_scalar(sans,y): #Loss function of scalar output
    a = abs(sans - y)
    if(a <= huber):
        loss = 0.5 * a * a
    else:
        loss = huber * a - 0.5 * huber * huber
    return loss

def loss_lookup(lans,y): #Loss function of lookup output
    L = 0
    for m in range(lans.shape[0]):
        for c in range(lans.shape[1]):
            L += y[m,c] * np.log(lans[m,c]) + (1 - y[m,c]) * np.log(1 - lans[m,c])
    L = L * (-1) / (lans.shape[0] * lans.shape[1])
    return L


data = generate_data(120,col_num)
d = len(cols[0])
W_col = Variable(torch.randn(d,2*d),requires_grad=True) #Initialize the weight matrix in data selector
W_op = Variable(torch.randn(d,2*d),requires_grad=True) #                              in operation selector 
W_Q = Variable(torch.randn(d,2*d),requires_grad=True) #                               in question RNN
W_H = Variable(torch.randn(d,3*d),requires_grad=True) #                               in history RNN


#generate the questions
q_3col = question_generate_threecol(data)
#q_2col = question_generate_twocol(data)
q_1col = question_generate_onecol(data)
q_arit = question_generate_arith(data)
all_q = []
all_q.extend(q_3col)
#all_q.extend(q_2col)
all_q.extend(q_1col)
all_q.extend(q_arit)
#allq = list(((set(q_3col)|set(q_2col))|set(q_1col))|set(q_arit))
rd.shuffle(all_q)
all_size = len(all_q)
test_data = all_q[:int(all_size/10)] #Split train data and test data
train_data = all_q[int(all_size/10):]
data_dict = {'train':train_data,'test':test_data}
with open('D:/Yifu/NP_data.pkl','wb')as f:
    pickle.dump(data_dict,f,pickle.HIGHEST_PROTOCOL)

#Load the existing data
#with open('D:/Yifu/NP_data.pkl','rb')as f:
#    data_dict = pickle.load(f)
#train_data = data_dict['train']
#test_data = data_dict['test']



step = 500 #Value of training steps
batch_size = 256 #Size of input in one epoch
losses = [] #Store the loss
for epoch in range(step):
    startnum = rd.randint(0,len(train_data) - batch_size)
    train_batch = train_data[startnum:startnum + batch_size] #Get one batch randomly
    L = torch.zeros(1)
    for (question,answer,type) in train_batch: #'type' is to know whether the output is scalar or lookup vector
        #print('1',end='')
        qrnn_input, piv_input = question_change(question)
        scalar_answers = [0]
        lookup_answers = [np.zeros((120, col_num))]
        row_selects = [np.ones((120, 1))]
        q_rep, hid_z = question_rnn_forward(qrnn_input,W_Q)
        ht = np.zeros((d,1))
        aop = np.zeros((16,1))
        acol = np.zeros((col_num,1))
        for time in range(4): #Four time steps
            ht = history_RNN_forward(aop,acol,W_H,ht,time,d)
            aop = selector_op_forward(W_op,q_rep,ht)
            acol = selector_data_forward(W_col,q_rep,ht)
            scalar_answers,lookup_answers,row_selects = operation_forward(data,scalar_answers,lookup_answers,row_selects,piv_input,aop,acol)
        if(type == 1): #Scalar Output
            result = scalar_answers[-1]
            L += torch.tensor(loss_scalar(result,answer)/batch_size,requires_grad=True)
        if(type == 0): #Lookup Output
            result = lookup_answers[-1]
            L += torch.tensor(loss_lookup(result,answer) * lamb/batch_size,requires_grad=True)
    L.backward() #BP
    W_col.data -= W_col.grad.data * alpha #Update the variables
    W_op.data -= W_op.grad.data * alpha
    W_H.data -= W_H.grad.data * alpha
    W_Q.data -= W_Q.grad.data * alpha
    op_mat.data -= op_mat.grad.data * alpha
    col_mat.data -= col_mat.grad.data * alpha
    W_col.grad.data.zero_()
    W_op.grad.data.zero_()
    W_H.grad.data.zero_()
    W_Q.grad.data.zero_()
    op_mat.grad.data.zero_()
    col_mat.grad.data.zero_()
    print(epoch,L.item())








