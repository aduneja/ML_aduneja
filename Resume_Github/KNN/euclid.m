function dist=euclid(x,y,m)
dist=0;
for i=1:m
    dist=dist+(((x(i)-y(i))^2));
end
dist=sqrt(dist);
