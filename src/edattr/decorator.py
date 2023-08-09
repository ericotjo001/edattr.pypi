######## components ########

def print_fname(func,kwargs,v=20):
    if 'verbose' in kwargs:
        if kwargs['verbose']>=v: 
            print(func.__qualname__)

def print_output(func_output, kwargs, v=100):
    if 'verbose' in kwargs:
        if kwargs['verbose']>=v: 
            print(func_output)

def prall_step(kwargs, f_out, v=200):
    if 'verbose' in kwargs: 
        steptext = ''
        if 'pstep' in kwargs:
            steptext += f' [{kwargs["pstep"]}]'
        if kwargs['verbose']>=v: 
            print(f'{steptext} {f_out}')   

def prsome_step(kwargs, f_out, v=200):
    do_print = False
    if 'verbose' in kwargs and 'pstep' in kwargs:
        if kwargs['verbose']>=v:
            do_print = True

    if not do_print: return

    steptext = ''
    if 'pstep' in kwargs:
        steptext += f' [{kwargs["pstep"]}]'
    
    if  kwargs["pstep"]<2:
        print(f'{steptext} {f_out}')
    if kwargs["pstep"]==2:
        print(f'  ...')

######## functions ########

def printfunc(func):
    def decorated_func(*args,**kwargs):
        print_fname(func, kwargs, v=20)
        func_output = func(*args, **kwargs)
        print_output(func_output, kwargs, v=100)
        return func_output
    return decorated_func

def printoutput(func):
    def decorated_func(*args,**kwargs):
        func_output = func(*args, **kwargs)
        print_output(func_output, kwargs, v=100)
        return func_output
    return decorated_func

def print_rall(func):
    # print recursive all
    def dec_func(*args, **kwargs):
        f_out = func(*args, **kwargs)
        prall_step(kwargs, f_out)
        return f_out
    return dec_func

def print_rsome(func):
    # print recursive some
    def dec_func(*args, **kwargs):
        f_out = func(*args, **kwargs)
        prsome_step(kwargs,f_out)
        return f_out
    return dec_func


