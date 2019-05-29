function dist=mod_euclid(x,y,m,info)
dist=0;
for i=1:m
    dist=dist+(((x(i)-y(i))/info(i))^2);
end
dist=sqrt(dist);
