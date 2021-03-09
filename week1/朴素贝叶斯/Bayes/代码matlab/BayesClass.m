classdef BayesClass
    properties   %定义属性
        cla_data
        classify
        test
        attribute
    end
    methods  %定义类的方法
        function obj = BayesClass(cla_data,classify,test,attribute) %构造函数,声明为obj（可以自己设定） 相当于self
            obj.cla_data=cla_data;
            obj.classify=classify;
            obj.test=test;
            obj.attribute=attribute;
        end
        function [mmean,mvar]= Mean_var(obj)  %[mean,var]是返回值，Mean_var函数名（obj 参数，obj代表class 的self）
            mmean=[];
            mvar=[];
            for i=1:size(obj.cla_data,1)
                for cls=1:size(obj.cla_data,2)%创建tuple，求tuple的大小
                    temp_mean=[];
                    temp_var=[];
                    for col=1:size(obj.cla_data(cls).data,2)
                        if obj.attribute(col)==1          
                            temp_mean=[temp_mean  mean(obj.cla_data(cls).data(:,col) ,1)];
                            temp_var=[temp_var std(obj.cla_data(cls).data(:,col))^2] ; 
                        end
                    end
                    mmean=[mmean ;temp_mean];
                    mvar=[mvar ;temp_var];
                end
            end
        end
        %计算条件概率(连续)
        function pro= Condition_pro(x,mean,var)
            pro=(1/(sqrt(2*pi)*var))*exp(-(power(x-mean,2))/(2*power(var,2)));
        end
        % 计算条件概率(离散)   
        function ans_pro=condition_dispe(obj,x,x_num,classify_num) 
            sum=size(obj.cla_data(classify_num).data,1);
            tmp=obj.cla_data(classify_num).data;
            num=size(tmp(tmp(:,x_num)==x),1);
            ans_pro=num/sum;          
        end
        %计算每个类的概率 
        function rate=class_pro(obj,sum)
            rate=[];
            for i=1:size(obj.cla_data,2)
                rate=[rate size(obj.cla_data(i).data,1)/sum];
            end
        end
        %预测    
        function Accuracy=forecast(obj,rate,mean,var)
            sum=size(obj.test,1);
            yes=0;
            size(obj.test,1)
            for i=1:size(obj.test,1)
                current=0;%记录当前概率最大值
                cur_classify="";%记录对应的预测分类结果
                for j=1:size(obj.classify)
                    temp=rate(j);
                    continuity=0; %记录第几个连续/离散数据
                    for attr=1:size(obj.attribute)-1
                        if obj.attribute(attr)==1%连续的
                            temp=temp*Condition_pro(obj.test(i,attr),mean(j,continuity),var(j,continuity));
                            continuity=continuity+1;
                        elseif obj.attribute(attr)==0 %离散的
                            temp=temp*condition_dispe(obj.test(i,attr),attr,j);
                        end
                    end
                    if temp>current
                        current=temp;
                        cur_classify=obj.classify(j);
                    end
                end
                tp=obj.test(i,size(obj.attribute));
                if cur_classify== tp(2)
                    yes=yes+1;
                end
                
            end  
            Accuracy=yes/sum
        end
     
        end
     
end



        
    