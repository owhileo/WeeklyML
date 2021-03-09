% %调用BayesClass.m文件的函数，此文件主要用于数据的处理
%将string 类型的数据处理成数字 用数字来表示
% %每次运行前都需要清空工作区，因为动态创建了struct
% 

% %读取带string的csv文件
% file_id=fopen("test_adult.txt")
% file=textscan(file_id,'%f%s%f%s%f%s%s%s%s%s%f%f%f%s%s','Delimiter',",")%'Delimiter'代表分隔符是，
% % attribute=[1,1,1,1,2];%( 0：离散 1：连续 2：classify)  iris数据集的
% attribute=[1,0,1,0,1,0,0,0,0,0,1,1,1,0,2];
% %将string 类型的数据处理成数字 用数字来表示
% for i=1:size(attribute,2)
%     if attribute(i)~=1 %不是连续的数据都是string 都需要变成数字
% %           struct_data=struct('name',string(file{i}{1}),'num',1);
%         struct_data.name=string(file{i}{1});  %动态创建字典,存储已经出现过的string
%         struct_data.num=1;%代表应该转换成的数字
%         type=1;%一共有几个类型的离散值
%         for j=1:size(file{i},1)
%             temp=[];
%             temp=[temp struct_data.name];  %存储name
%             temp
%             flag=0;%代表没有出现过此string，其他都代表应该转换成的数字
%             for t =1:size(temp,2) %遍历字典
%                 if string(temp(t))==string(file{i}{j})
%                     flag=t;
%                     continue;                    
%                 end
%             end
%             if flag ~=0   %不是新的name
%                 file{i}{j}=flag;
%             else    %新的name
%                 type=type+1;
%                 struct_data(type).name=string(file{i}{j});
%                 struct_data(type).num=type;
%                 file{i}{j}=type;
%             end
%             string(file{i}{j})
%         end
%     end
%     struct_data={}
% end
%数据的访问
% file{1}(1) %访问连续属性
% file{2}{1} %访问非连续属性
% size(file{2},1) %7505
% 将离散值处理后为cell的部分 变成double类型
% for i=1:size(file,2)
%     if iscell(file{i})==1
%         file{i}=cell2mat(file{i})
%     end
% end

% %将file变成double类型的数组
% file=cell2mat(file);

% %按类生成数组
% classify=2; %有几个classify
% d=struct('name',[],'data',[]);
% for i =1:classify
%     cla_data=file(file(:,15)==i,:);
%     d(i).name=i;
%     d(i).data=cla_data;
% end

%访问
% d(1).name  
% hh=d(1).data %double类型的数据
% mean(d(1).data(:,2))
classify=[1,2];
attribute=[1,0,1,0,1,0,0,0,0,0,1,1,1,0,2];
Adult=BayesClass(d,classify,file,attribute);
rate= Adult.class_pro(7505);
meanvar=Adult.Mean_var();
result= Adult.forecast(rate,meanvar(1,:),meanvar(2,:));



























% %动态创建struct 数组    !!!!!!学习！！！！-------------------------------------
% Student.name = "wangx";
% Student.sex = 'Male';
% 
% Student(2).name = "hh";
% Student(2).sex = 'Male';
% 
% fieldnames(Student);%name,sex
% temp=[]
% temp=[temp Student.name] %"wangx"    "hh"  !!!声明的时候”“ 和’‘的区别!!!!
% temp=[]
% temp=[temp Student.sex] % 'MaleMale'


























% classify={"Iris-setosa","Iris-versicolor","Iris-virginica"};
% 
% 
% data_test=convert("test.txt",attribute);
% data_train=convert_classify("train.txt",attribute,classify);
% 
% 
% % %输出内容：
% % data_test{1};
% % % 1×5 cell 数组
% % %     {[4.9000]}    {[3]}    {[1.4000]}    {[0.2000]}    {'Iris-setosa'}
% % 
% % data_test{1}{5} %'Iris-setosa'
% % 
% % data_test{1}{:}
% % % 4.9000
% % % 3
% % % 1.4000
% % % 0.2000
% % % 'Iris-setosa'
% 
% data_train{1,3};  %可以读取数据  {[6.3000]}    {[3.3000]}    {[6]}    {[2.5000]}    {'Iris-virginica'}
% hh=size(data_train,1);
% t=[];
% for i=1:39
%     t=[t ;data_train{i,1}]
% end
% 
% 
% %把数据集处理成tuple形式的数据
% function data= convert(file,attribute)
%     f=importdata(file);
%     for i=1:size(f,1)
%         temp= strsplit( f{i},',');
%         for j =1:size(temp,2)
%             if attribute(j)==1
%                 t{j}=str2num(temp{j});
%             else
%                 t{j}=temp{j};            
%             end   
%         end
%         data{i,:}=t;   %tuple的行的添加 
%     end
% end
% 
% %tuple={} 这样tuple{1}=1 不行 tuple={}就声明了其为空 不能动态添加
% 
% %把数据集按类处理成tuple形式的数据，每个类含一个tuple
% function data= convert_classify(file,attribute,classify)
%     f=importdata(file);
%     a=1;
%     b=1;
%     c=1;
%     for i=1:size(f,1)
%         temp= strsplit( f{i},',');
%         for j =1:size(temp,2)
%             if attribute(j)==1
%                 t{j}=str2num(temp{j});
%             else
%                 t{j}=temp{j};            
%             end   
%         end
%         if temp{size(temp,2)}== classify{1}
%             data{a,1}=t;
%             a=a+1;
%         elseif  temp{size(temp,2)}== classify{2}
%             data{b,2}=t;
%             b=b+1;
%         else
%             data{c,3}=t;
%             c=c+1;
%         end
%     end
% end
% 
% 
