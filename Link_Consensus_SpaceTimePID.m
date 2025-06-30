function [Opt_rank,Vs,M]=Link_Consensus_SpaceTimePID(X,Type)

%%X is a cell array of adjacency matrices
%%Type: 1 = Spatial, 2 = Temporal

%%Outputs:
        %Opt_rank: The number of modules identified that can be used as the input parameter for dimensionality reduction
        %Vs: The co-membership tensor
        %M: The final community partition vector
        
if Type==1
    Vs=[];
    Ms=[];
    for i=1:length(X)
        try
            M=link_communities(X{i},'single');
            Ms=[Ms;size(M,1)];
            for ii=1:size(M,1)
                Vs=cat(3,Vs,M(ii,:)'.*M(ii,:));
            end
        catch message
        end

    end

    M=link_communities(sum(Vs,3),'complete');
    Opt_rank=size(M,1);
elseif Type==2



    Vs=[];
    combos_s=nchoosek(1:size(X{1},1),2);
    for i=1:length(X)
        v=[];
        net=X{i};
        [M,Q]=community_louvain(net);
        for ii=1:length(combos_s)
            m1=M(combos_s(ii,1),:);
            m2=M(combos_s(ii,2),:);
            if isequal(m1,m2)
                v=[v;1];
            else
                v=[v;0];
            end
        end
        Vs=cat(3,Vs,squareform(v));
    end
    Vs=sum(Vs,3);

    [M,Q]=community_louvain(sum(Vs,3));
    Opt_rank=max(M);
end


end
