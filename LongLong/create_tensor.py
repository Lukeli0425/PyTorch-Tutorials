# 2022-2-9 luke
# demo of creating tensor with pytorch
import numpy as np
import torch

if __name__ == "__main__":
    # import from numpy
    a = np.array([2,3.3])
    b = torch.from_numpy(a)
    print(f"\n{a}\n{b}\n")

    a = np.ones([2,3])
    b = torch.from_numpy(a)
    print(f"\n{a}\n{b}\n")

    # import from list
    a = torch.tensor([2.,3.2])
    b = torch.FloatTensor([2.,3.2]) # input list to transform list to tensor
    c = torch.FloatTensor(2,3) # input integers to create a tensor of a specific shape
    d = torch.tensor([[2.,3.2],[1.,22.3]])
    print(f"\n{a}\n{b}\n{c}\n{d}\n")

    # create uninitialized tensor (asking for memory)
    a = torch.empty(1)
    b = torch.empty(2,3)
    c = torch.IntTensor(2,3)
    d = torch.FloatTensor(2,3)
    print(f"\n{a}\n{b}\n{c}\n{d}\n")

    # set default type
    a = torch.tensor(1.)
    torch.set_default_dtype(torch.float16)
    b = torch.tensor(1.)
    torch.set_default_dtype(torch.float64)
    c = torch.tensor([2.,3])
    torch.set_default_dtype(torch.float32)
    d = torch.tensor([1,0.])
    torch.set_default_dtype(torch.float32)
    print(f"\n{a.type()}\t{a.dtype}\n{b.type()}\t{b.dtype}")
    print(f"{c.type()}\t{c.dtype}\n{d.type()}\t{d.dtype}\n")

    # random initialization
    a = torch.rand(3,3) # create a random tensor of a specific shape
    b = torch.rand_like(a) # create a random tensor of a.shape
    c = torch.randint(1,10,[3,3]) # given shape, maximum & minimum create a random int tensor
    d = torch.randint_like(c,1,10) # given maximum & minimum create a random int tensor of a.shape
    print(f"\n{a}\n{b}\n{c}\n{d}\n")

    # random normarl initialization
    a = torch.randn(3,3) # N(0,1)
    b = torch.randn_like(a)
    c = torch.normal(mean=torch.full([10],0),std=torch.arange(1,0,-0.1))
    print(f"\n{a}\n{b}\n{c}\n")
