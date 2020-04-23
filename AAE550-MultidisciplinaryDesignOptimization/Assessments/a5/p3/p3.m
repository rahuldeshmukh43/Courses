%  assesment 5 P3: GA, V12

clc; clear all;

lb=[-4;2;-2;-2];
ub=[3;9;1;3];
r=1E-5;

% Part A: bits for x1 st distance btw decoded values is at most r
b=ceil(log((ub-lb)/r+1)/log(2))
% part B
l=sum(b)
% part C
Npop=4*l
% part D
Pm=(l+1)/(2*Npop*l)


% Integer Progtramming problem 
f=@(x) x(2)^2-7*x(1);

xi=1:1:7;

encoding= containers.Map;
encoding('000')=0;
encoding('001')=1;
encoding('010')=2;
encoding('011')=3;
encoding('100')=4;
encoding('101')=5;
encoding('110')=6;
encoding('111')=7;

A='011110';
B='100110';
C='101110';
D='010110';
parent_map=containers.Map({A,B,C,D},{'A','B','C','D'});

p=[0.91,0.16,0.46,0.42,0.42,0.77];
% p=kron(p,ones(1,100));

% Part E: tournament A&B and C&D
win1=tournament(A,B,encoding,f);
parent_map(win1)
win2=tournament(C,D,encoding,f);
parent_map(win2)

% Part F: crossover
Pc=0.5;
[C1,C2]=crossover(win1,win2,p,Pc)

