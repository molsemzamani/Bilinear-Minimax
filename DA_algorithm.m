%%%%%%%%%%% The example for the paper: Convergence rate analysis of the
%%%%%%%%%%% gradient descent-ascent method for convex-concave saddle-point
%%%%%%%%%%% problems
clear
clc

%%% Number of instances
NN=100;
%%% Number of iterations
N=6;

for pp=1:NN

    fname = sprintf('data%d.mat', pp);
    load(fname,'Ax','Ay','Axy','x0','y0')
    
    Mx=min(eig(Ax));
    My=min(eig(Ay));
    Lx=norm(Ax);
    Ly=norm(Ay);
    Lxy=norm(Axy);
    L1=max(Lx,Ly);
    M=min(Mx,My);
    L2=max([Lx,Ly,Lxy]);

    t1=(2*((L1+M)*sqrt(Lxy^2+L1*M)+Lxy*(M-L1)))/((4*Lxy^2+(L1+M)^2)*sqrt(Lxy^2+L1*M));
    t2=M/(4*L2^2);

    x2=x0;
    y2=y0;
    x1=x0;
    y1=y0;

    %%% Algorithm
    for kk=2:N+1
        xx=x1(:,kk-1)-t1*(Ax*x1(:,kk-1)+Axy*y1(:,kk-1));
        x1(:,kk)=xx;
        yy=y1(:,kk-1)+t1*(-Ay*y1(:,kk-1)+(x1(:,kk-1)'*Axy)');
        y1(:,kk)=yy;

        xx=x2(:,kk-1)-t2*(Ax*x2(:,kk-1)+Axy*y2(:,kk-1));
        x2(:,kk)=xx;
        yy=y2(:,kk-1)+t2*(-Ay*y2(:,kk-1)+(x2(:,kk-1)'*Axy)');
        y2(:,kk)=yy;

    end

    %%%% Calculating the distances
    pp1=[];
    pp2=[];
    for kk=1:N+1
        pp1=[pp1 norm(x1(:,kk))^2+norm(y1(:,kk))^2];
        pp2=[pp2 norm(x2(:,kk))^2+norm(y2(:,kk))^2];
    end

    PP1(pp,:)=pp1;
    PP2(pp,:)=pp2;

end

%%%% Calculating mean
for pp=1:N+1
    Mpp1(pp)=mean(PP1(:,pp));
    Mpp2(pp)=mean(PP2(:,pp));
end

%%% Plot
kk=0:N;
plot(kk,Mpp1,'DisplayName','t given by (11)')
hold on
plot(kk,Mpp2,'DisplayName','t=\mu/(4L^2)')
legend
xlabel('Iteration (k)') 
ylabel('Mean value of ||x^k-x^*||^2+||y^k-y^*||^2') 
title('Mean values  of ||x^k-x^*||^2+||y^k-y^*||^2 for 100 randomly generated instances for each iteration k using the two different step lengths')
set(gca,'xtick',0:N+1)
hold off
