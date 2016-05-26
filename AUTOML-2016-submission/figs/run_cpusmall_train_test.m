clear
addpath ../source/
addpath /home/bremen/Matlab_Projects/liblinear-2.1/matlab


for r=1:5

    %name='cpusmall';
    %load ~/Matlab_Projects/datasets/mat_files/cpusmall.mat
    %array_eta=2.^[-5:10];
    
    name='cadata';
    load /home/bremen/Matlab_Projects/datasets/mat_files/cadata
    array_eta=2.^[4:23];
    
    %name='yearPredictionMSD';
    %load YearPredictionMSD;
    %p=sparse(p);
    %array_eta=2.^[-5:5];

    % hp.type='rbf';
    % hp.gamma=1e-12;
    % K=compute_kernel(p(:,1:3000),p(:,1:3000),hp);
    % hp.gamma=hp.gamma/mean(-log(K(:)));
    % clear K;
    t=t-mean(t);
    %t=t/std(t);

    [p,t]=shuffledata(p,t);

    for i=1:size(p,2)
        p(:,i)=p(:,i)/norm(p(:,i));
    end

    n=size(p,2);
    n_tr=round(n*0.75);

    pt=p(:,n_tr:end);
    tt=t(n_tr:end);
    p=p(:,1:n_tr);
    t=t(1:n_tr);

    model_bak=model_init();
    model_kt=kt_absloss_train(p,t,model_bak);
    [tmp,preds]=model_predict(pt,model_kt,0);
    err_kt(r)=mean(abs(preds-tt));
    [tmp,preds]=model_predict(pt,model_kt,1);
    err_kt_av(r)=mean(abs(preds-tt));

    model_eg=dfeg_abs_train(p,t,model_bak);
    [tmp,preds]=model_predict(pt,model_eg,0);
    err_eg(r)=mean(abs(preds-tt))
    [tmp,preds]=model_predict(pt,model_eg,1);
    err_eg_av(r)=mean(abs(preds-tt))

    model_eg_new=dfeg_new_abs_train(p,t,model_bak);
    [tmp,preds]=model_predict(pt,model_eg_new,0);
    err_eg_new(r)=mean(abs(preds-tt))
    [tmp,preds]=model_predict(pt,model_eg_new,1);
    err_eg_new_av(r)=mean(abs(preds-tt))

    %model_adagrad=adagrad_rda_sql2_diag_train(p,t,model_bak);
    %model_betting=cocob_absloss_train(p,t,model_bak);
    model_pistol=pistol_abs_train(p,t,model_bak);
    [tmp,preds]=model_predict(pt,model_pistol,0);
    err_pistol(r)=mean(abs(preds-tt))
    [tmp,preds]=model_predict(pt,model_pistol,1);
    err_pistol_av(r)=mean(abs(preds-tt))

%         %array_eta=2.^[-10:20];
%         array_eta=2.^[-1:1];
%         for i=1:numel(array_eta)
%             err(i)=train(t',p',sprintf('-s 13 -c %d -v 5',array_eta(i)));
%         end
%         [mn,idx_mn]=min(err);
%         model_svm=train(t',p',sprintf('-s 13 -c %d',array_eta(idx_mn)));
%         preds=predict(tt',pt',model_svm);
%         err_svm(idx_perc,r)=mean(abs(preds-tt'));

    for i=1:numel(array_eta)
        model_bak.eta=array_eta(i);
        model_gd{i}=gd_abs_train(p,t,model_bak);
        err_gd(r,i)=mean(abs(tt-model_gd{i}.w*pt/sqrt(model_gd{i}.iter)));
        err_gd_av(r,i)=mean(abs(tt-model_gd{i}.w2*pt/model_gd{i}.iter));
    end
end
n=size(p,2);

clear p
clear t
save([name '_experiment_train_test.mat']);

if 1
load([name '_experiment_train_test.mat']);
close all
set(0,'DefaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
h1=ploterr(array_eta,mean(err_gd),[],std(err_gd),'logx')
set(h1(1),'Color','k'), set(h1(2),'Color','k')
hold on
%plot(perc*n_tr,err_eg,'y')
%plot(perc*n_tr,err_eg_new,'g')
%plot(perc,err_pistol,'b')
%semilogx([array_eta],zeros(size([array_eta]))+model_betting.ael(end)*size(p,2),'m')
h2=ploterr(array_eta,zeros(1,numel(array_eta))+mean(err_kt),[],std(err_kt),'logx')
set(h2(1),'Color','r'), set(h2(2),'Color','r')
%h3=ploterr(array_eta,zeros(1,numel(array_eta))+mean(err_kt_av),[],std(err_kt_av),'logx')
%set(h3(1),'Color','g'), set(h3(2),'Color','g')
%h4=ploterr(array_eta,zeros(1,numel(array_eta))+mean(err_gd_av),[],std(err_gd_av),'logx')
%set(h4(1),'Color','m'), set(h4(2),'Color','m')
%legend('DFEG, NIPS 2014','Adaptive Normal, COLT 2014','PiSTOL, NIPS 2014', 'COCOB, Draft 2015', 'Kernel GD, various eta','location', 'NorthEast')
%legend('DFEG, NIPS 2014','Adaptive Normal, COLT 2014', 'COCOB, Draft 2015', 'Kernel GD, various eta','location', 'NorthEast')
h=legend([h1(1) h2(1)],'SGD','KT-based','location', 'Best');
set(h,'Interpreter','latex');
ylabel('Test loss')
xlabel('Size training set')
a=axis;
a(1)=array_eta(1)/2;
a(2)=array_eta(end)*2;
%a(4)=11.5;
axis(a)
title([name ' dataset, absolute loss'])
grid on
print( gcf, '-dpdf', [name '_kt_train_test.pdf'])
end