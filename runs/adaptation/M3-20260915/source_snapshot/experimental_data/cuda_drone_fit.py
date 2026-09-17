"""GPU replay of the existing weighted translation fitting recurrence."""
import torch
from .current_adaptation_fit import WeightedTranslationBatch
from simulator.cuda_autograd import CudaAutogradBlock


class CudaDroneFit:
    def __init__(self,trials,parameters,network):
        batch=WeightedTranslationBatch(trials,parameters,device='cuda')
        self.batch=batch
        kp,kd,ff,_=parameters.tensors(batch.p)
        indices=[torch.as_tensor(plan[2],device='cuda') for plan in batch.plans]
        weights=[batch.p.new_tensor(t.weights[:len(y)]) for t,y in zip(trials,batch.truth)]
        def objective(p,v):
            ps=[p];penalty=p.new_zeros(len(p))
            for h,c in zip(batch.dt,batch.cmd):
                extra=network(p,v,batch.bias,c)
                a=kp*(c[:,:3]-p)+kd*(c[:,3:6]-v)+ff*c[:,6:9]+batch.bias+extra
                pm=p+.5*h*v;vm=v+.5*h*a
                delta=network(pm,vm,batch.bias,c)
                am=kp*(c[:,:3]-pm)+kd*(c[:,3:6]-vm)+ff*c[:,6:9]+batch.bias+delta
                p=p+h*vm;v=v+h*am;ps.append(p)
                penalty=penalty+h[:,0]*delta.square().mean(-1)
            allp=torch.stack(ps);losses=[]
            for i,(index,w,y) in enumerate(zip(indices,weights,batch.truth)):
                p=allp[index,i];target=torch.where(w[:,None]>0,y,p.detach())
                sq=(p-target).square().sum(-1)/.05**2
                losses.append((2*(torch.sqrt(1+sq)-1)*w).sum())
            regularizer=(penalty/batch.dt[:,:,0].sum(0)).mean()
            return (torch.stack(losses).mean()+.01*regularizer/.5**2,)
        self.block=CudaAutogradBlock(objective,(batch.p,batch.v),tuple(network.parameters()))

    def __call__(self):return self.block(self.batch.p,self.batch.v)[0]
