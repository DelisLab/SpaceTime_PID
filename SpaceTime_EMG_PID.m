function [R_spacetime,S_spacetime,UYZ_spacetime,net_R_space,net_S_space,net_UYZ_space,net_R_time,net_S_time,net_UYZ_time] = SpaceTime_EMG_PID(EMG,TASK,window_length)


%%Input:
        %EMG: Tensor [No. of timepoints x No. of Muscles x No. of trials]
        %TASK: Tensor [No. of timepoints x No. of parameters x No. of trials]

%%Output
        %R_spacetime,S_spacetime,UYZ_spacetime = Final output matrix for dimensionality reduction
        %net_R_space,net_S_space,net_UYZ_space = Spatial networks for community detection
        %net_R_time,net_S_time,net_UYZ_time = Temporal networks for community detection


EMG=mat2tiles(EMG,window_length);EMG=cat(4,EMG{:});
TASK=mat2tiles(TASK,window_length);TASK=cat(4,TASK{:});

combos_s=nchoosek(1:size(EMG,2),2);
combos_t=[nchoosek(1:size(EMG,4),2);[1:size(EMG,4);1:size(EMG,4)]'];

R=[];S=[];UY=[];UZ=[];
for i=1:length(combos_s)
    for ii=1:length(combos_t)
        for c=1:size(TASK,2)

            emg_x=EMG(:,combos_s(i,1),:,combos_t(ii,1));emg_y=EMG(:,combos_s(i,2),:,combos_t(ii,2));
           
            task=TASK(:,c,:,combos_t(ii,2));%task2=TASK(:,c,:,combos_t(ii,2));

            try
                [r,s,uy,uz] = Gaussian_PID(emg_x(:),emg_y(:),task(:));
                R=[R;r];S=[S;s];UY=[UY;uy];UZ=[UZ;uz];
            catch message
                R=[R;0];S=[S;0];UY=[UY;0];UZ=[UZ;0];
            end
        end
    end
end
R=permute(reshape(R,[size(TASK,2),length(combos_t),length(combos_s)]),[3,2,1]);
S=permute(reshape(S,[size(TASK,2),length(combos_t),length(combos_s)]),[3,2,1]);
UY=permute(reshape(UY,[size(TASK,2),length(combos_t),length(combos_s)]),[3,2,1]);
UZ=permute(reshape(UZ,[size(TASK,2),length(combos_t),length(combos_s)]),[3,2,1]);

R_space=[];S_space=[];UY_space=[];UZ_space=[];
net_R_space={};net_S_space={};net_UYZ_space={};
mask=tril(true(size(zeros(size(EMG,2),size(EMG,2)))),-1);
for c=1:size(TASK,2)
    for ii=1:length(combos_t)

        A=squareform(R(1:length(combos_s),ii,c));
        [threshold] = modified_percolation_analysis(A);A(A<threshold)=0;A(A<0)=0;
        net_R_space=cat(2,net_R_space,A);
        R_space=cat(2,R_space,A(mask));

        A=squareform(S(1:length(combos_s),ii,c));
        [threshold] = modified_percolation_analysis(A);A(A<threshold)=0;A(A<0)=0;
        net_S_space=cat(2,net_S_space,A);
        S_space=cat(2,S_space,A(mask));

        A=squareform(UY(1:length(combos_s),ii,c));
        [threshold] = modified_percolation_analysis(A);A(A<threshold)=0;A(A<0)=0;
        net_UYZ_space=cat(2,net_UYZ_space,A);
        UY_space=cat(2,UY_space,A(mask));

        A=squareform(UZ(1:length(combos_s),ii,c));
        [threshold] = modified_percolation_analysis(A);A(A<threshold)=0;A(A<0)=0;
        net_UYZ_space=cat(2,net_UYZ_space,A);
        UZ_space=cat(2,UZ_space,A(mask));
    end
end
R_space=reshape(R_space,[length(combos_s),length(combos_t),size(TASK,2)]);
S_space=reshape(S_space,[length(combos_s),length(combos_t),size(TASK,2)]);
UY_space=reshape(UY_space,[length(combos_s),length(combos_t),size(TASK,2)]);
UZ_space=reshape(UZ_space,[length(combos_s),length(combos_t),size(TASK,2)]);

R_time=[];S_time=[];UY_time=[];UZ_time=[];
net_R_time={};net_S_time={};net_UYZ_time={};
mask=tril(true(size(zeros(size(EMG,4),size(EMG,4)))),-1);
for c=1:size(TASK,2)
    for ii=1:length(combos_s)
    
        A=squareform(R(ii,1:length(combos_t)-size(EMG,4),c))+diag(R(ii,[length(combos_t)-size(EMG,4)]+1:end,c));
        [threshold] = modified_percolation_analysis(A);A(A<threshold)=0;A(A<0)=0;
        net_R_time=cat(2,net_R_time,A);
        R_time=cat(2,R_time,[A(mask);diag(A)]);

        A=squareform(S(ii,1:length(combos_t)-size(EMG,4),c))+diag(S(ii,[length(combos_t)-size(EMG,4)]+1:end,c));
        [threshold] = modified_percolation_analysis(A);A(A<threshold)=0;A(A<0)=0;
        net_S_time=cat(2,net_S_time,A);
        S_time=cat(2,S_time,[A(mask);diag(A)]);

        A=squareform(UY(ii,1:length(combos_t)-size(EMG,4),c))+diag(UY(ii,[length(combos_t)-size(EMG,4)]+1:end,c));
        [threshold] = modified_percolation_analysis(A);A(A<threshold)=0;A(A<0)=0;
        net_UYZ_time=cat(2,net_UYZ_time,A);
        UY_time=cat(2,UY_time,[A(mask);diag(A)]);

        A=squareform(UZ(ii,1:length(combos_t)-size(EMG,4),c))+diag(UZ(ii,[length(combos_t)-size(EMG,4)]+1:end,c));
        [threshold] = modified_percolation_analysis(A);A(A<threshold)=0;A(A<0)=0;
        net_UYZ_time=cat(2,net_UYZ_time,A);
        UZ_time=cat(2,UZ_time,[A(mask);diag(A)]);
    end
end
R_time=reshape(R_time,[length(combos_t),length(combos_s),size(TASK,2)]);
S_time=reshape(S_time,[length(combos_t),length(combos_s),size(TASK,2)]);
UY_time=reshape(UY_time,[length(combos_t),length(combos_s),size(TASK,2)]);
UZ_time=reshape(UZ_time,[length(combos_t),length(combos_s),size(TASK,2)]);

R_spacetime=zeros([length(combos_t),length(combos_s),size(TASK,2)]);
S_spacetime=zeros([length(combos_t),length(combos_s),size(TASK,2)]);
UY_spacetime=zeros([length(combos_t),length(combos_s),size(TASK,2)]);
UZ_spacetime=zeros([length(combos_t),length(combos_s),size(TASK,2)]);
for c = 1:size(TASK,2)
    for i = 1:length(combos_s)
        for ii = 1:length(combos_t)
            % R
            if R_time(ii,i,c) ~= 0 && R_space(i,ii,c)
                R_spacetime(ii,i,c) = R_time(ii,i,c);
            end
            % S
            if S_time(ii,i,c) ~= 0 && S_space(i,ii,c)
                S_spacetime(ii,i,c) = S_time(ii,i,c);
            end
            % UY
            if UY_time(ii,i,c) ~= 0 && UY_space(i,ii,c)
                UY_spacetime(ii,i,c) = UY_time(ii,i,c);
            end
            % UZ
            if UZ_time(ii,i,c) ~= 0 && UZ_space(i,ii,c)
                UZ_spacetime(ii,i,c) = UZ_time(ii,i,c);
            end
        end
    end
end
UYZ_spacetime=cat(1,UY_spacetime,UZ_spacetime);

net_R_space={};net_S_space={};net_UYZ_space={};
for c=1:size(R_spacetime,3)
    for ii=1:size(R_spacetime,1)
        A=squareform(R_spacetime(ii,:,c));
        net_R_space=cat(2,net_R_space,A);
        A=squareform(S_spacetime(ii,:,c));
        net_S_space=cat(2,net_S_space,A);
        A=squareform(UYZ_spacetime(ii,:,c));
        net_UYZ_space=cat(2,net_UYZ_space,A);
        A=squareform(UYZ_spacetime(ii+length(combos_t),:,c));
        net_UYZ_space=cat(2,net_UYZ_space,A);
    end
end

end