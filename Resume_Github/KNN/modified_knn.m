function val= modified_knn(traindata,testvec,k)
 
info=ig(traindata);
n=size(traindata,2);
tr_m=size(traindata,1);%number of rows
d=zeros(tr_m,1);% dis a m by 1 vector
arr=zeros(1,k);% array is a row vector
tr_x=traindata(:,1:n-1); %the input matrix
tr_y=traindata(:,n);%last column in the dataset should be the class vector
%te_m=size(testdata,1);
for i=1:tr_m
    d(i)=mod_euclid(testvec,tr_x(i,:),n-1,info); %find the euclidean distance from the testvector to every training example
end
tr_x=[d tr_x tr_y]; %merge the vectors
for i=1:k
   z= find(tr_x(:,1)==min(tr_x(:,1)));
  if(length(z)>1)
   arr(i)=tr_y(z(1));
  else
      arr(i)=tr_y(z);
  end
    tr_x(z,:)=[];                            %find the class of the training example which has the least euclidean distance
end                                             %repeat this ktimes
tbl=tabulate(arr);
ind=find(tbl(:,2)==max(tbl(:,2)));           %find the most frequent class and assign that class as the class of the test vector
val=tbl(ind,1);
 
 



