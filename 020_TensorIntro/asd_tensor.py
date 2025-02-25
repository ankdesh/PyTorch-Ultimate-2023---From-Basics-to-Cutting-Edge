import torch

x = torch.tensor(100.0, requires_grad=True)
y = x * x 
print (y)
print (y.backward())
print (x.grad)

a = torch.tensor(100.0, requires_grad=True)

def func(x):
    y = x*x + 2*x + 1
    return y

print (func (a))

asd = func(a).backward()

y = x*x + 2*x + 1
print (y)
y.backward()
print (y.grad)
print (x.grad)


a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2

print (Q)

print (a.grad)

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

print (a.grad)
print (b.grad)

p = torch.tensor(10., requires_grad=True)
q = p**2
r = q**3

print (r)
r.backward()

print (q.grad)
print (p.grad)

print (p.requires_grad)
print (q.requires_grad)
print (r.requires_grad)

# Storing intermediate gradient too
p = torch.tensor(10., requires_grad=True)
q = p**2
r = q**3

q.retain_grad()

print (r)
r.backward()

print (q.grad)
print (p.grad)
