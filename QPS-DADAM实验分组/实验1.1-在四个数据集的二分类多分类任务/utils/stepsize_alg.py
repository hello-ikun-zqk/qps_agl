import math

def get_stepsize(step_init,iters_num,method='decay-sq',lambda_val=0.1):
    if method=='decay-sq':
        step=step_init/(1+step_init*lambda_val*iters_num)**0.5
    elif method=='decay':
        step=step_init/(1+step_init*lambda_val*iters_num)
    elif method=='decay-2':
        step=step_init/(1+iters_num)
    elif method=='decay-3':
        step=step_init/(lambda_val+iters_num)
    elif method=='decay-0.05':
        step=step_init/(1+step_init*lambda_val*iters_num)**0.05
    elif method=='fix':
        step=step_init
    elif method=='frac_sqrt_t':
        step=1/(2*math.sqrt(iters_num+1))
    else:
        raise Exception("No such step size method!")

    return step


def get_beta1t(beta1,iters_num,lambda_val=0.9):
    return beta1*lambda_val**(iters_num-1)