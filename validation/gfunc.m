function [gt]= gfunc(n, D, a, b, u, l, t, tL,tU)
gt=0;

for i=1:n     
    
    tli=(a(i)-l(i)*D(i))/b(i);
    tui=(a(i)-u(i)*D(i))/b(i);
    
    if (t<=tui)
        gt=gt+b(i)*u(i);
    else if ((tui<=t)&&(t<=tli))
        gt=gt+b(i)*(a(i)-t*b(i))/D(i);
    else if (tli<=t)
        gt=gt+b(i)*l(i);
        end
        end
    end
end

end