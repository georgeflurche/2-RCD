function [xtstar]=kiwiel(n,r,D,a,b,u,l)


%Kiwiel Algorithm for continuous quadratic knapsack problem
% Knapsack problem:
%     min 0.5 x'Dx-a'x
%     s.t.: b'x=r, l<=x<=u
% Matrix D=diag(d), with d>0 (convex objective function)
% K.C. Kiwiel,
% On Linear-Time Algorithms for the Continuous Quadratic Knapsack Problem,
% J. Optim Theory Appl (2007) 134: 549–554.
% where: 
%     n - scalar: dimension
%     r - scalar
%     D - array of size nx1
%     a - array of size nx1
%     b - array of size nx1
%     u - array of size nx1 (upper limit)
%     l - array of size nx1 (lower limit)


%Step 0 : Initializare

dim=2*n; Ind=[1:n]; lold = l;
Ind = Ind(b<0);
a = diag(sign(b))*a;
l(b<0) = -u(b<0); 
u(b<0) = -lold(b<0);
b = diag(sign(b))*b;


for i=1:n
    tl(i)=(a(i)-l(i)*D(i))/b(i);
    tu(i)=(a(i)-u(i)*D(i))/b(i);
    T(i)=tl(i);                   
    T(i+n)=tu(i);
end

tL=tu(1);
tU=tl(1);
tL = min(tu);
tU = max(tl);

while (dim>0)
    %Step 1. Median of T;
    aux=rand(1);
    ran=ceil(aux*dim);
    tmed1=T(ran);

    %Step 2. Computing g(t)
    gt=gfunc(n,D,a,b,u,l,tmed1,tL,tU);

    %Step 3
    if (gt==r)%((gt+eqprec>r)&&(gt-eqprec<r))
        tstar=tmed1;
        %[tmed1 tl' tu']
        %xtstar=zeros(n,1);
        disp('S-a apelat break\n');
        break;
    end
    
    %Step 4
    if (gt>r)
       tL=tmed1;
       T = T(T>tmed1);
       dimnou = length(T);
       dim=dimnou; 

    else  %Step 5 

        tU=tmed1;
        T = T(T<tmed1);
        dimnou = length(T);
        dim=dimnou;

    end
    p=0;
    q=0;
    s=0;
    for i=1:n

        if (tl(i)<=tL)
            s=s+b(i)*l(i);
        end    
        if (tu(i)>=tU)
            s=s+b(i)*u(i);
        end
        if ((tu(i)<=tL)&&(tL<=tl(i))&&(tu(i)<=tU)&&(tU<=tl(i)))
            p=p+a(i)*b(i)/D(i);
            q=q+b(i)*b(i)/D(i);
        end
    end
    tstar=(p+s-r)/q;
    disp([p s r q, tstar]);
end
xtstar = zeros(n,1);
for i=1:n
    if (tstar<=tu(i))
        xtstar(i)=u(i);   
    elseif ((tu(i)<=tstar)&&(tstar<=tl(i)))
        xtstar(i)=(a(i)-tstar*b(i))/D(i);
    elseif (tl(i)<=tstar)
        xtstar(i)=l(i);   
    end       
end


xtstar(Ind) = -xtstar(Ind);

end
