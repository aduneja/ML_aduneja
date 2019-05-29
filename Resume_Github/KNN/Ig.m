function info=ig(tr_data)
m=size(tr_data,1);
n=size(tr_data,2);%no of attributes
y=tr_data(:,n);%taking the last column as the decision class
arr=unique(y);%unique classes are in an array
num_classes=length(unique(y));%no of classes(decisions)
num_div_attr=zeros(n-1,1);                                                                                                                                                                                    
for i=1:n-1
    num_div_attr(i)=length(unique(tr_data(:,i))); %no of divisions of each attribute in a decison tree
end
res_mat=zeros(num_classes,max(num_div_attr),n-1);
for i=1:n-1
    temp=tr_data(:,i); %first choose an attribute whch is going to be a separate matrix
    div=unique(temp); %find the unique divisions of that attribute and store it in an array
    for j=1:num_div_attr(i)   %loop till the no of divisions as it is going to be the columns part of each attribute matrix
        c=ismember(temp,div(j));
        indexes=find(c) ;         %check which of the indices of that attribute contains that division
        for k=1:length(indexes)  %loop till the no of times a division appears in an attribute
            cl_index=find(arr==y(indexes(k))); %find the corresponding class attribute for that division
            res_mat(cl_index,j,i)=res_mat(cl_index,j,i)+1; %fill the matrix
        end
    end
    
end
res_mat;
num_div_attr;
%now we are calculating the entropy
entset=zeros(n-1,1);
addr=0;
for i=1:n-1                    % we have to calculate the entropy for every attribute so loopin from 1 to n
  var=num_div_attr(i);
    s=res_mat(:,:,i);
    s1=s(:,[1 size(s,2)]);
   col_sum=sum(s1,2);
   for j=1:size(s1,2)
       for k=1:num_classes
           addr=addr+((-res_mat(k,j)*log10(res_mat(k,j))/col_sum(j)));
       end
     entset(i)=entset(i)+((col_sum(j)/m)*addr);  
   end
end
%calculating initial entropy
entr_ini=0;
for i=1:num_classes
    c=ismember(arr(i),y);%might have to change it
    indexes=find(c); %might have to change it
    v=length(indexes);
    entr_ini=entr_ini+((-v/m)*log(v/m));
end
%calculating information gain of each attibute
%info=zeros(n,1);
info=entr_ini-entset;
 
