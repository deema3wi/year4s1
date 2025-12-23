function [trainIdx, testIdx] = ida_holdout_split(N, trainRatio, seed)
rng(seed);
idx = randperm(N);
Ntrain = round(trainRatio * N);
trainIdx = idx(1:Ntrain);
testIdx  = idx(Ntrain+1:end);
end
