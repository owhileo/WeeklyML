% %����BayesClass.m�ļ��ĺ��������ļ���Ҫ�������ݵĴ���
%��string ���͵����ݴ�������� ����������ʾ
% %ÿ������ǰ����Ҫ��չ���������Ϊ��̬������struct
% 

% %��ȡ��string��csv�ļ�
% file_id=fopen("test_adult.txt")
% file=textscan(file_id,'%f%s%f%s%f%s%s%s%s%s%f%f%f%s%s','Delimiter',",")%'Delimiter'����ָ����ǣ�
% % attribute=[1,1,1,1,2];%( 0����ɢ 1������ 2��classify)  iris���ݼ���
% attribute=[1,0,1,0,1,0,0,0,0,0,1,1,1,0,2];
% %��string ���͵����ݴ�������� ����������ʾ
% for i=1:size(attribute,2)
%     if attribute(i)~=1 %�������������ݶ���string ����Ҫ�������
% %           struct_data=struct('name',string(file{i}{1}),'num',1);
%         struct_data.name=string(file{i}{1});  %��̬�����ֵ�,�洢�Ѿ����ֹ���string
%         struct_data.num=1;%����Ӧ��ת���ɵ�����
%         type=1;%һ���м������͵���ɢֵ
%         for j=1:size(file{i},1)
%             temp=[];
%             temp=[temp struct_data.name];  %�洢name
%             temp
%             flag=0;%����û�г��ֹ���string������������Ӧ��ת���ɵ�����
%             for t =1:size(temp,2) %�����ֵ�
%                 if string(temp(t))==string(file{i}{j})
%                     flag=t;
%                     continue;                    
%                 end
%             end
%             if flag ~=0   %�����µ�name
%                 file{i}{j}=flag;
%             else    %�µ�name
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
%���ݵķ���
% file{1}(1) %������������
% file{2}{1} %���ʷ���������
% size(file{2},1) %7505
% ����ɢֵ�����Ϊcell�Ĳ��� ���double����
% for i=1:size(file,2)
%     if iscell(file{i})==1
%         file{i}=cell2mat(file{i})
%     end
% end

% %��file���double���͵�����
% file=cell2mat(file);

% %������������
% classify=2; %�м���classify
% d=struct('name',[],'data',[]);
% for i =1:classify
%     cla_data=file(file(:,15)==i,:);
%     d(i).name=i;
%     d(i).data=cla_data;
% end

%����
% d(1).name  
% hh=d(1).data %double���͵�����
% mean(d(1).data(:,2))
classify=[1,2];
attribute=[1,0,1,0,1,0,0,0,0,0,1,1,1,0,2];
Adult=BayesClass(d,classify,file,attribute);
rate= Adult.class_pro(7505);
meanvar=Adult.Mean_var();
result= Adult.forecast(rate,meanvar(1,:),meanvar(2,:));



























% %��̬����struct ����    !!!!!!ѧϰ��������-------------------------------------
% Student.name = "wangx";
% Student.sex = 'Male';
% 
% Student(2).name = "hh";
% Student(2).sex = 'Male';
% 
% fieldnames(Student);%name,sex
% temp=[]
% temp=[temp Student.name] %"wangx"    "hh"  !!!������ʱ�򡱡� �͡���������!!!!
% temp=[]
% temp=[temp Student.sex] % 'MaleMale'


























% classify={"Iris-setosa","Iris-versicolor","Iris-virginica"};
% 
% 
% data_test=convert("test.txt",attribute);
% data_train=convert_classify("train.txt",attribute,classify);
% 
% 
% % %������ݣ�
% % data_test{1};
% % % 1��5 cell ����
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
% data_train{1,3};  %���Զ�ȡ����  {[6.3000]}    {[3.3000]}    {[6]}    {[2.5000]}    {'Iris-virginica'}
% hh=size(data_train,1);
% t=[];
% for i=1:39
%     t=[t ;data_train{i,1}]
% end
% 
% 
% %�����ݼ������tuple��ʽ������
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
%         data{i,:}=t;   %tuple���е���� 
%     end
% end
% 
% %tuple={} ����tuple{1}=1 ���� tuple={}����������Ϊ�� ���ܶ�̬���
% 
% %�����ݼ����ദ���tuple��ʽ�����ݣ�ÿ���ຬһ��tuple
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
