
%% Perturbational foundation-model simulation
% Original sandbox for simulation purposes, by Hamidreza Jamalabadi.
% Saves simulation results and figures to ./results/matlab/.

clear; close all; clc;
fprintf('Running publication script v2 (TIFF compatibility fix)\n');

%% Reproducibility and output
cfg.seed = 11;
rng(cfg.seed, 'twister');

cfg.outputDir = fullfile(pwd, 'results', 'matlab');
if ~exist(cfg.outputDir, 'dir')
    mkdir(cfg.outputDir);
end

%% Parallel execution
cfg.useParallel = true;
cfg.maxWorkers = [];
cfg.startPoolAutomatically = true;

%% Core simulation settings
cfg.dt = 0.02;

cfg.trainSubjects = 60;
cfg.testSubjects = 20;
cfg.trainStepsPerSubject = 1500;
cfg.testSteps = 1400;
cfg.burnIn = 150;

cfg.calibrationSizes = [2 3 5 8 12 20 35 60 100];
cfg.learningCurveYScale = 'log';
cfg.calibrationEpisodes = 2;

%% Ground-truth subject family
cfg.aRange = [1.00 1.00];
cfg.cRange = [0.75 1.30];
cfg.dRange = [-0.28 0.28];
cfg.bRange = [0.60 1.45];
cfg.sigmaRange = [0.14 0.14];

%% Population-pretraining perturbations
cfg.trainPulseProbability = 0.035;
cfg.trainPulseDurationRange = [35 90];
cfg.trainPulseAmplitudeRange = [-2.30 2.30];
cfg.trainInputNoiseSD = 0.10;
cfg.randomiseInitialBasin = true;

%% New-subject calibration perturbations
cfg.calPulseCount = 18;
cfg.calPulseDurationRange = [30 75];
cfg.calPulseAmplitudeRange = [-1.70 1.70];
cfg.calInputNoiseSD = 0.08;
cfg.calPerturbedFraction = 0.50;

%% Held-out test perturbation
cfg.testPulseStart = 430;
cfg.testPulseDuration = 120;
cfg.testPulseAmplitude = 2.25;
cfg.testAmplitudeGrid = linspace(0.5, 2.8, 12);

% Distributional and structural evaluation
cfg.flowInputGrid = unique([-2.0 -1.0 0 1.0 2.0 cfg.testPulseAmplitude]);
cfg.densityEdges = linspace(-2.3, 2.3, 81);
cfg.invariantSteps = 5000;
cfg.invariantBurnIn = 1000;
cfg.distributionReplicates = 40;
cfg.responseEvalStride = 20;
cfg.densityPseudoCount = 1e-8;

%% Training target
cfg.targetMode = 'trueDrift';

%% Landscape grid
cfg.xGrid = linspace(-1.9, 1.9, 450);

%% Shared neural model
net.hidden1 = 24;
net.hidden2 = 24;
net.embeddingDim = 3;
net.inputDim = 2 + net.embeddingDim;

%% Optimisation
opt.pretrainEpochs = 55;
opt.pretrainBatchSize = 2048;
opt.pretrainLearningRate = 2e-3;
opt.adaptationEpochs = 120;
opt.embeddingLearningRate = 3e-2;
opt.weightDecay = 1e-5;
opt.embeddingPenalty = 1e-3;
opt.gradientClip = 5.0;

%% Plotting colors
plt.groundTruth = [0.05 0.05 0.05];
plt.scratch = [0.00 0.447 0.698];       % colour-blind-safe blue
plt.passive = [0.835 0.369 0.000];       % colour-blind-safe orange
plt.perturb = [0.000 0.620 0.451];       % colour-blind-safe green
plt.pulseShade = [0.82 0.82 0.82];
plt.populationAlpha = 0.55;

fprintf('Generating population data...\n');

%% Training population
trainParams = sampleSubjectParameters(cfg.trainSubjects, cfg);

passiveData = generatePopulationDataset( ...
    trainParams, cfg.trainStepsPerSubject, cfg, false);

perturbData = generatePopulationDataset( ...
    trainParams, cfg.trainStepsPerSubject, cfg, true);

fprintf('Passive samples: %d\n', numel(passiveData.x));
fprintf('Perturbational samples: %d\n', numel(perturbData.x));

%% Standardisation
scale.xMean = mean(perturbData.x);
scale.xStd = std(perturbData.x) + eps;
scale.uMean = mean(perturbData.u);
scale.uStd = std(perturbData.u) + eps;
scale.yMean = mean(perturbData.y);
scale.yStd = std(perturbData.y) + eps;

passiveData = standardiseDataset(passiveData, scale);
perturbData = standardiseDataset(perturbData, scale);

%% Shared-model training
baseModel = initialiseSharedModel(net, cfg.trainSubjects);

fprintf('\nTraining passive population model...\n');
passiveModel = trainSharedModel(baseModel, passiveData, opt, 'Passive');

fprintf('\nTraining perturbational population model...\n');
perturbModel = trainSharedModel(baseModel, perturbData, opt, 'Perturbational');

%% Unseen-subject evaluation
testParams = sampleSubjectParameters(cfg.testSubjects, cfg);

nT = cfg.testSubjects;
nC = numel(cfg.calibrationSizes);
nM = 3;

passiveRMSE = nan(nT, nC, nM);
perturbRMSE = nan(nT, nC, nM);
landscapeRMSE = nan(nT, nC, nM);
barrierError = nan(nT, nC, nM);
attractorError = nan(nT, nC, nM);
transitionCorrect = nan(nT, nC, nM);
amplitudeCurveRMSE = nan(nT, nC, nM);
flowRMSE = nan(nT, nC, nM);
invariantKL = nan(nT, nC, nM);
responseJS = nan(nT, nC, nM);
transitionProbabilityRMSE = nan(nT, nC, nM);
coverageStats = nan(nT, 5);
exampleCells = cell(nT,1);

exampleSubject = 1;
exampleCalibrationIndex = 6;  % 20 samples

useParfor = configureParallelPool(cfg);

fprintf('\nEvaluating unseen subjects...\n');

if useParfor
    parfor s = 1:nT
        [passiveRMSE(s,:,:), perturbRMSE(s,:,:), ...
         landscapeRMSE(s,:,:), barrierError(s,:,:), ...
         attractorError(s,:,:), transitionCorrect(s,:,:), ...
         amplitudeCurveRMSE(s,:,:), flowRMSE(s,:,:), ...
         invariantKL(s,:,:), responseJS(s,:,:), ...
         transitionProbabilityRMSE(s,:,:), coverageStats(s,:), ...
         exampleCells{s}] = evaluateOneSubject( ...
            s, testParams(s), passiveModel, perturbModel, ...
            cfg, opt, scale, exampleCalibrationIndex);
    end
else
    for s = 1:nT
        [passiveRMSE(s,:,:), perturbRMSE(s,:,:), ...
         landscapeRMSE(s,:,:), barrierError(s,:,:), ...
         attractorError(s,:,:), transitionCorrect(s,:,:), ...
         amplitudeCurveRMSE(s,:,:), flowRMSE(s,:,:), ...
         invariantKL(s,:,:), responseJS(s,:,:), ...
         transitionProbabilityRMSE(s,:,:), coverageStats(s,:), ...
         exampleCells{s}] = evaluateOneSubject( ...
            s, testParams(s), passiveModel, perturbModel, ...
            cfg, opt, scale, exampleCalibrationIndex);

        fprintf('  completed subject %d/%d\n', s, nT);
    end
end

results.passiveRMSE = passiveRMSE;
results.perturbRMSE = perturbRMSE;
results.landscapeRMSE = landscapeRMSE;
results.barrierError = barrierError;
results.attractorError = attractorError;
results.transitionCorrect = transitionCorrect;
results.amplitudeCurveRMSE = amplitudeCurveRMSE;
results.flowRMSE = flowRMSE;
results.invariantKL = invariantKL;
results.responseJS = responseJS;
results.transitionProbabilityRMSE = transitionProbabilityRMSE;
results.coverageStats = coverageStats;

example = exampleCells{exampleSubject};
summary = summariseResults(results, cfg);

save(fullfile(cfg.outputDir, 'simulation_results.mat'), ...
    'cfg', 'opt', 'net', 'trainParams', 'testParams', ...
    'passiveModel', 'perturbModel', 'results', 'summary', ...
    'example', 'scale', 'plt', '-v7.3');

makeMainFigure(cfg, trainParams, summary, example, plt);
makeSupplementFigure(cfg, summary, example, plt);
makeCoverageFigure(cfg, passiveData, perturbData, results, plt);
writeSummaryTable(cfg, summary);

fprintf('\nFinished. Results saved in:\n%s\n', cfg.outputDir);

%% ============================ LOCAL FUNCTIONS ============================

function useParfor = configureParallelPool(cfg)
useParfor = false;

if ~cfg.useParallel
    fprintf('Parallel execution disabled.\n');
    return;
end

if ~license('test', 'Distrib_Computing_Toolbox')
    fprintf('Parallel Computing Toolbox unavailable; using FOR.\n');
    return;
end

try
    pool = gcp('nocreate');

    if isempty(pool) && cfg.startPoolAutomatically
        if isempty(cfg.maxWorkers)
            pool = parpool('local');
        else
            pool = parpool('local', cfg.maxWorkers);
        end
    end

    if ~isempty(pool)
        fprintf('Using PARFOR with %d workers.\n', pool.NumWorkers);
        useParfor = true;
    end
catch ME
    warning('Parallel pool could not be started: %s', ME.message);
end
end

function [subPassiveRMSE, subPerturbRMSE, subLandscapeRMSE, ...
          subBarrierError, subAttractorError, subTransitionCorrect, ...
          subAmplitudeCurveRMSE, subFlowRMSE, subInvariantKL, ...
          subResponseJS, subTransitionProbabilityRMSE, subCoverage, example] = ...
          evaluateOneSubject(subjectIndex, p, passiveModel, perturbModel, ...
          cfg, opt, scale, exampleCalibrationIndex)

nC = numel(cfg.calibrationSizes);
nM = 3;

subPassiveRMSE = nan(1,nC,nM);
subPerturbRMSE = nan(1,nC,nM);
subLandscapeRMSE = nan(1,nC,nM);
subBarrierError = nan(1,nC,nM);
subAttractorError = nan(1,nC,nM);
subTransitionCorrect = nan(1,nC,nM);
subAmplitudeCurveRMSE = nan(1,nC,nM);
subFlowRMSE = nan(1,nC,nM);
subInvariantKL = nan(1,nC,nM);
subResponseJS = nan(1,nC,nM);
subTransitionProbabilityRMSE = nan(1,nC,nM);
subCoverage = nan(1,5);
example = struct();

rng(1000 + subjectIndex, 'twister');

allX = [];
allU = [];
allY = [];
episodeTrajectories = cell(cfg.calibrationEpisodes,1);
episodeInputs = cell(cfg.calibrationEpisodes,1);

for ep = 1:cfg.calibrationEpisodes
    uCal = makeCalibrationInput(cfg.testSteps, cfg);
    initialBasin = chooseInitialBasin(ep);
    xCal = simulateSubject(p, uCal, cfg, initialBasin);

    xNow = xCal(1:end-1);
    uNow = uCal(1:end-1);

    if strcmpi(cfg.targetMode, 'trueDrift')
        yNow = trueDrift(xNow, uNow, p);
    else
        yNow = diff(xCal) / cfg.dt;
    end

    allX = [allX; xNow]; %#ok<AGROW>
    allU = [allU; uNow]; %#ok<AGROW>
    allY = [allY; yNow]; %#ok<AGROW>

    episodeTrajectories{ep} = xCal;
    episodeInputs{ep} = uCal;
end

trueV = truePotential(cfg.xGrid, p);
trueV = trueV - min(trueV);
trueFeatures = landscapeFeatures(cfg.xGrid, trueV);

% Ground-truth controlled vector field on a common state-input grid
[flowX, flowU] = ndgrid(cfg.xGrid, cfg.flowInputGrid);
trueFlow = trueDrift(flowX, flowU, p);

% Ground-truth passive invariant distribution and held-out response ensemble
trueInvariantSamples = simulateStochasticTrueEnsemble( ...
    p, zeros(cfg.invariantSteps,1), cfg, initialStateFromParams(p), ...
    cfg.distributionReplicates, 31000 + subjectIndex);
trueInvariantProb = empiricalProbability( ...
    trueInvariantSamples(cfg.invariantBurnIn+1:end,:), cfg.densityEdges, cfg.densityPseudoCount);

uResponse = zeros(cfg.testSteps,1);
responseStop = min(cfg.testSteps, cfg.testPulseStart + cfg.testPulseDuration - 1);
uResponse(cfg.testPulseStart:responseStop) = cfg.testPulseAmplitude;
trueResponseEnsemble = simulateStochasticTrueEnsemble( ...
    p, uResponse, cfg, initialStateFromParams(p), ...
    cfg.distributionReplicates, 32000 + subjectIndex);

subCoverage(1) = mean(allX < trueFeatures.saddleX - 0.15);
subCoverage(2) = mean(abs(allX - trueFeatures.saddleX) <= 0.15);
subCoverage(3) = mean(allX > trueFeatures.saddleX + 0.15);
subCoverage(4) = mean(abs(allU) > 0.15);
subCoverage(5) = countBasinTransitions(allX, trueFeatures.saddleX);

uPassive = zeros(cfg.testSteps,1);
uTest = zeros(cfg.testSteps,1);
testStop = min(cfg.testSteps, ...
    cfg.testPulseStart + cfg.testPulseDuration - 1);
uTest(cfg.testPulseStart:testStop) = cfg.testPulseAmplitude;

initialState = -sqrt(max(p.c/p.a,0.05));

rng(10000 + subjectIndex, 'twister');
xPassiveTrue = simulateSubject(p, uPassive, cfg, initialState);

rng(20000 + subjectIndex, 'twister');
xPerturbTrue = simulateSubject(p, uTest, cfg, initialState);

for ci = 1:nC
    nCal = min(cfg.calibrationSizes(ci), numel(allX));

    idx = sampleCalibrationIndices( ...
        allU, nCal, cfg.calPerturbedFraction, ...
        5000 + 100*subjectIndex + ci);

    cal.x = allX(idx);
    cal.u = allU(idx);
    cal.y = allY(idx);
    cal.subject = ones(numel(idx),1);
    cal = standardiseDataset(cal, scale);

    ePassive = adaptEmbedding(passiveModel, cal, opt);
    ePerturb = adaptEmbedding(perturbModel, cal, opt);
    cubicModel = fitCubicScratchModel(cal);

    predPassiveSet = { ...
        rolloutCubic(cubicModel, initialState, uPassive, cfg, scale), ...
        rolloutShared(passiveModel, ePassive, initialState, uPassive, cfg, scale), ...
        rolloutShared(perturbModel, ePerturb, initialState, uPassive, cfg, scale)};

    predPerturbSet = { ...
        rolloutCubic(cubicModel, initialState, uTest, cfg, scale), ...
        rolloutShared(passiveModel, ePassive, initialState, uTest, cfg, scale), ...
        rolloutShared(perturbModel, ePerturb, initialState, uTest, cfg, scale)};

    Vset = { ...
        recoverCubicPotential(cubicModel, cfg.xGrid, scale), ...
        recoverSharedPotential(passiveModel, ePassive, cfg.xGrid, scale), ...
        recoverSharedPotential(perturbModel, ePerturb, cfg.xGrid, scale)};

    trueCurve = computeTransitionCurveTrue(p, cfg, initialState);
    predCurves = { ...
        computeTransitionCurveCubic(cubicModel, cfg, scale, initialState), ...
        computeTransitionCurveShared(passiveModel, ePassive, cfg, scale, initialState), ...
        computeTransitionCurveShared(perturbModel, ePerturb, cfg, scale, initialState)};

    for m = 1:nM
        subPassiveRMSE(1,ci,m) = ...
            sqrt(mean((predPassiveSet{m} - xPassiveTrue).^2));

        subPerturbRMSE(1,ci,m) = ...
            sqrt(mean((predPerturbSet{m} - xPerturbTrue).^2));

        Vhat = alignPotentialOffsetOnly(Vset{m});

        subLandscapeRMSE(1,ci,m) = ...
            sqrt(mean((Vhat - trueV).^2));

        estimatedFeatures = landscapeFeatures(cfg.xGrid, Vhat);

        subBarrierError(1,ci,m) = ...
            abs(estimatedFeatures.barrier - trueFeatures.barrier);

        subAttractorError(1,ci,m) = matchedAttractorError( ...
            estimatedFeatures.minimaX, trueFeatures.minimaX);

        trueTransition = basinTransition(xPerturbTrue, trueFeatures.saddleX);
        predTransition = basinTransition(predPerturbSet{m}, trueFeatures.saddleX);

        subTransitionCorrect(1,ci,m) = ...
            double(trueTransition == predTransition);

        subAmplitudeCurveRMSE(1,ci,m) = ...
            sqrt(mean((predCurves{m} - trueCurve).^2));

        % 1) Dynamical structure: controlled vector-field error
        if m == 1
            estimatedFlow = cubicPredictDrift(cubicModel, flowX(:), flowU(:), scale);
        elseif m == 2
            estimatedFlow = sharedPredictDrift(passiveModel, ePassive, flowX(:), flowU(:), scale);
        else
            estimatedFlow = sharedPredictDrift(perturbModel, ePerturb, flowX(:), flowU(:), scale);
        end
        estimatedFlow = reshape(estimatedFlow, size(trueFlow));
        subFlowRMSE(1,ci,m) = sqrt(mean((estimatedFlow(:)-trueFlow(:)).^2));

        % 2) Dynamical structure: KL divergence of passive invariant occupancy
        zeroLong = zeros(cfg.invariantSteps,1);
        if m == 1
            modelInvariant = simulateStochasticCubicEnsemble( ...
                cubicModel, zeroLong, cfg, scale, initialState, ...
                cfg.distributionReplicates, p.sigma, 41000 + 1000*subjectIndex + 100*ci + m);
        elseif m == 2
            modelInvariant = simulateStochasticSharedEnsemble( ...
                passiveModel, ePassive, zeroLong, cfg, scale, initialState, ...
                cfg.distributionReplicates, p.sigma, 41000 + 1000*subjectIndex + 100*ci + m);
        else
            modelInvariant = simulateStochasticSharedEnsemble( ...
                perturbModel, ePerturb, zeroLong, cfg, scale, initialState, ...
                cfg.distributionReplicates, p.sigma, 41000 + 1000*subjectIndex + 100*ci + m);
        end
        modelInvariantProb = empiricalProbability( ...
            modelInvariant(cfg.invariantBurnIn+1:end,:), cfg.densityEdges, cfg.densityPseudoCount);
        subInvariantKL(1,ci,m) = klDivergence(trueInvariantProb, modelInvariantProb);

        % 3) Perturbational response: distributional divergence over time
        if m == 1
            modelResponse = simulateStochasticCubicEnsemble( ...
                cubicModel, uResponse, cfg, scale, initialState, ...
                cfg.distributionReplicates, p.sigma, 51000 + 1000*subjectIndex + 100*ci + m);
        elseif m == 2
            modelResponse = simulateStochasticSharedEnsemble( ...
                passiveModel, ePassive, uResponse, cfg, scale, initialState, ...
                cfg.distributionReplicates, p.sigma, 51000 + 1000*subjectIndex + 100*ci + m);
        else
            modelResponse = simulateStochasticSharedEnsemble( ...
                perturbModel, ePerturb, uResponse, cfg, scale, initialState, ...
                cfg.distributionReplicates, p.sigma, 51000 + 1000*subjectIndex + 100*ci + m);
        end
        subResponseJS(1,ci,m) = meanResponseJSDivergence( ...
            trueResponseEnsemble, modelResponse, cfg.densityEdges, ...
            cfg.responseEvalStride, cfg.densityPseudoCount);

        % 4) Intervention outcome: stochastic dose-transition probabilities
        trueProbCurve = computeTransitionProbabilityCurveTrue( ...
            p, cfg, initialState, cfg.distributionReplicates, ...
            61000 + 1000*subjectIndex + 100*ci);
        if m == 1
            modelProbCurve = computeTransitionProbabilityCurveCubic( ...
                cubicModel, cfg, scale, initialState, p.sigma, ...
                cfg.distributionReplicates, 71000 + 1000*subjectIndex + 100*ci + m);
        elseif m == 2
            modelProbCurve = computeTransitionProbabilityCurveShared( ...
                passiveModel, ePassive, cfg, scale, initialState, p.sigma, ...
                cfg.distributionReplicates, 71000 + 1000*subjectIndex + 100*ci + m);
        else
            modelProbCurve = computeTransitionProbabilityCurveShared( ...
                perturbModel, ePerturb, cfg, scale, initialState, p.sigma, ...
                cfg.distributionReplicates, 71000 + 1000*subjectIndex + 100*ci + m);
        end
        subTransitionProbabilityRMSE(1,ci,m) = ...
            sqrt(mean((modelProbCurve-trueProbCurve).^2));
    end

    if ci == exampleCalibrationIndex
        example.params = p;
        example.calibrationX = episodeTrajectories{1};
        example.calibrationU = episodeInputs{1};
        example.truePerturb = xPerturbTrue;
        example.predPerturb = predPerturbSet;
        example.Vtrue = trueV;
        example.Vset = { ...
            alignPotentialOffsetOnly(Vset{1}), ...
            alignPotentialOffsetOnly(Vset{2}), ...
            alignPotentialOffsetOnly(Vset{3})};
        example.trueFeatures = trueFeatures;
        example.nCal = nCal;
        example.trueCurve = trueCurve;
        example.predCurves = predCurves;
        example.trueFlow = trueFlow;
        example.flowX = flowX;
        example.flowU = flowU;
        example.trueInvariantProb = trueInvariantProb;
        example.trueResponseEnsemble = trueResponseEnsemble;
        example.trueProbCurve = trueProbCurve;
        example.modelProbCurves = { ...
            computeTransitionProbabilityCurveCubic(cubicModel,cfg,scale,initialState,p.sigma,cfg.distributionReplicates,81001), ...
            computeTransitionProbabilityCurveShared(passiveModel,ePassive,cfg,scale,initialState,p.sigma,cfg.distributionReplicates,81002), ...
            computeTransitionProbabilityCurveShared(perturbModel,ePerturb,cfg,scale,initialState,p.sigma,cfg.distributionReplicates,81003)};
        example.modelInvariantSamples = { ...
            simulateStochasticCubicEnsemble(cubicModel,zeros(cfg.invariantSteps,1),cfg,scale,initialState,cfg.distributionReplicates,p.sigma,83001), ...
            simulateStochasticSharedEnsemble(passiveModel,ePassive,zeros(cfg.invariantSteps,1),cfg,scale,initialState,cfg.distributionReplicates,p.sigma,83002), ...
            simulateStochasticSharedEnsemble(perturbModel,ePerturb,zeros(cfg.invariantSteps,1),cfg,scale,initialState,cfg.distributionReplicates,p.sigma,83003)};
        example.flowPredictions = { ...
            reshape(cubicPredictDrift(cubicModel,flowX(:),flowU(:),scale),size(flowX)), ...
            reshape(sharedPredictDrift(passiveModel,ePassive,flowX(:),flowU(:),scale),size(flowX)), ...
            reshape(sharedPredictDrift(perturbModel,ePerturb,flowX(:),flowU(:),scale),size(flowX))};
        example.modelResponseEnsembles = { ...
            simulateStochasticCubicEnsemble(cubicModel,uResponse,cfg,scale,initialState,cfg.distributionReplicates,p.sigma,82001), ...
            simulateStochasticSharedEnsemble(passiveModel,ePassive,uResponse,cfg,scale,initialState,cfg.distributionReplicates,p.sigma,82002), ...
            simulateStochasticSharedEnsemble(perturbModel,ePerturb,uResponse,cfg,scale,initialState,cfg.distributionReplicates,p.sigma,82003)};
        example.selectedX = allX(idx);
        example.selectedU = allU(idx);
        example.selectedIdx = idx;
    end
end
end

function basin = chooseInitialBasin(ep)
if mod(ep,2)==1
    basin = -1;
else
    basin = 1;
end
end

function idx = sampleCalibrationIndices(u, nCal, perturbedFraction, seed)
rng(seed, 'twister');

perturbed = find(abs(u) > 0.15);
unperturbed = find(abs(u) <= 0.15);

nPert = min(round(nCal * perturbedFraction), numel(perturbed));
nUnpert = min(nCal - nPert, numel(unperturbed));

idxPert = [];
idxUnpert = [];

if nPert > 0
    idxPert = perturbed(randperm(numel(perturbed), nPert));
end

if nUnpert > 0
    idxUnpert = unperturbed(randperm(numel(unperturbed), nUnpert));
end

idx = [idxPert(:); idxUnpert(:)];

if numel(idx) < nCal
    remaining = setdiff((1:numel(u))', idx);
    addN = min(nCal-numel(idx), numel(remaining));
    idx = [idx; remaining(randperm(numel(remaining), addN))];
end

idx = sort(idx);
end

function params = sampleSubjectParameters(n, cfg)
params = repmat(struct('a',[],'c',[],'d',[],'b',[],'sigma',[]), n,1);
for i = 1:n
    params(i).a = uniformSample(cfg.aRange);
    params(i).c = uniformSample(cfg.cRange);
    params(i).d = uniformSample(cfg.dRange);
    params(i).b = uniformSample(cfg.bRange);
    params(i).sigma = uniformSample(cfg.sigmaRange);
end
end

function value = uniformSample(range)
if range(1)==range(2)
    value = range(1);
else
    value = range(1) + rand()*diff(range);
end
end

function data = generatePopulationDataset(params, stepsPerSubject, cfg, withPerturbations)
nSubjects = numel(params);
samplesPerSubject = stepsPerSubject - 1;
totalSamples = nSubjects * samplesPerSubject;

allX = zeros(totalSamples,1);
allU = zeros(totalSamples,1);
allY = zeros(totalSamples,1);
allSubject = zeros(totalSamples,1);

cursor = 0;

for s = 1:nSubjects
    totalSteps = stepsPerSubject + cfg.burnIn;

    if withPerturbations
        u = makeRandomPulseInput(totalSteps, cfg);
    else
        u = zeros(totalSteps,1);
    end

    if cfg.randomiseInitialBasin
        initialBasin = 2*(rand>0.5)-1;
    else
        initialBasin = -1;
    end

    x = simulateSubject(params(s), u, cfg, initialBasin);
    x = x(cfg.burnIn+1:end);
    u = u(cfg.burnIn+1:end);

    xNow = x(1:end-1);
    uNow = u(1:end-1);

    if strcmpi(cfg.targetMode, 'trueDrift')
        yNow = trueDrift(xNow, uNow, params(s));
    else
        yNow = diff(x) / cfg.dt;
    end

    idx = cursor + (1:samplesPerSubject);

    allX(idx) = xNow;
    allU(idx) = uNow;
    allY(idx) = yNow;
    allSubject(idx) = s;

    cursor = cursor + samplesPerSubject;
end

data.x = allX;
data.u = allU;
data.y = allY;
data.subject = allSubject;
end

function u = makeRandomPulseInput(T, cfg)
u = cfg.trainInputNoiseSD * randn(T,1);
t = 1;

while t <= T
    if rand < cfg.trainPulseProbability
        duration = randi(cfg.trainPulseDurationRange);
        amplitude = cfg.trainPulseAmplitudeRange(1) + rand()*diff(cfg.trainPulseAmplitudeRange);

        stopIdx = min(T, t+duration-1);
        u(t:stopIdx) = u(t:stopIdx) + amplitude;
        t = stopIdx + 1;
    else
        t = t + 1;
    end
end

u = max(min(u, max(abs(cfg.trainPulseAmplitudeRange))), -max(abs(cfg.trainPulseAmplitudeRange)));
end

function u = makeCalibrationInput(T, cfg)
u = cfg.calInputNoiseSD * randn(T,1);

candidateStarts = randperm(max(1,T-100), min(cfg.calPulseCount, max(1,T-100)));

for k = 1:numel(candidateStarts)
    st = candidateStarts(k);
    duration = randi(cfg.calPulseDurationRange);
    amplitude = cfg.calPulseAmplitudeRange(1) + rand()*diff(cfg.calPulseAmplitudeRange);
    en = min(T, st+duration-1);
    u(st:en) = u(st:en) + amplitude;
end

limit = max(abs(cfg.calPulseAmplitudeRange));
u = max(min(u, limit), -limit);
end

function x = simulateSubject(p, u, cfg, initialBasin)
T = numel(u);
x = zeros(T,1);

base = sqrt(max(p.c/p.a,0.05));

if initialBasin < 0
    x(1) = -base + 0.08*randn();
elseif initialBasin > 0
    x(1) = base + 0.08*randn();
else
    x(1) = 0.15*randn();
end

for t = 1:T-1
    drift = trueDrift(x(t), u(t), p);
    x(t+1) = x(t) + cfg.dt*drift + p.sigma*sqrt(cfg.dt)*randn();
    x(t+1) = max(min(x(t+1),2.3),-2.3);
end
end

function drift = trueDrift(x,u,p)
drift = -(p.a.*x.^3 - p.c.*x - p.d) + p.b.*u;
end

function V = truePotential(x,p)
V = p.a.*x.^4./4 - p.c.*x.^2./2 - p.d.*x;
end

function data = standardiseDataset(data, scale)
data.xs = (data.x-scale.xMean)/scale.xStd;
data.us = (data.u-scale.uMean)/scale.uStd;
data.ys = (data.y-scale.yMean)/scale.yStd;
end

function model = initialiseSharedModel(net,nSubjects)
r = 0.10;
model.W1 = r*randn(net.hidden1,net.inputDim);
model.b1 = zeros(net.hidden1,1);
model.W2 = r*randn(net.hidden2,net.hidden1);
model.b2 = zeros(net.hidden2,1);
model.W3 = r*randn(1,net.hidden2);
model.b3 = 0;
model.E = 0.05*randn(net.embeddingDim,nSubjects);
model = resetAdamState(model);
end

function model = resetAdamState(model)
names = {'W1','b1','W2','b2','W3','b3','E'};
for i = 1:numel(names)
    name = names{i};
    model.adamM.(name) = zeros(size(model.(name)));
    model.adamV.(name) = zeros(size(model.(name)));
end
model.adamStep = 0;
end

function model = trainSharedModel(model,data,opt,label)
n = numel(data.xs);
nBatches = ceil(n/opt.pretrainBatchSize);

for epoch = 1:opt.pretrainEpochs
    order = randperm(n);
    epochLoss = 0;

    for b = 1:nBatches
        startIdx = (b-1)*opt.pretrainBatchSize+1;
        stopIdx = min(b*opt.pretrainBatchSize,n);
        idx = order(startIdx:stopIdx);

        x = data.xs(idx)';
        u = data.us(idx)';
        y = data.ys(idx)';
        subjects = data.subject(idx)';

        [loss,grads] = sharedLossAndGradients(model,x,u,y,subjects,opt);
        grads = clipGradients(grads,opt.gradientClip);
        model = adamUpdate(model,grads,opt.pretrainLearningRate);

        epochLoss = epochLoss + loss*numel(idx);
    end

    if epoch==1 || mod(epoch,5)==0 || epoch==opt.pretrainEpochs
        fprintf('  %s epoch %3d/%3d | loss %.6f\n', label, epoch, opt.pretrainEpochs, epochLoss/n);
    end
end
end

function [loss,grads] = sharedLossAndGradients(model,x,u,y,subjects,opt)
Ebatch = model.E(:,subjects);
X = [x;u;Ebatch];

H1 = tanh(model.W1*X + model.b1);
H2 = tanh(model.W2*H1 + model.b2);
Yhat = model.W3*H2 + model.b3;

n = size(X,2);
err = Yhat-y;

loss = mean(err.^2) + ...
    opt.weightDecay*(sum(model.W1(:).^2)+sum(model.W2(:).^2)+sum(model.W3(:).^2)) + ...
    opt.embeddingPenalty*mean(Ebatch(:).^2);

dY = 2*err/n;

grads.W3 = dY*H2' + 2*opt.weightDecay*model.W3;
grads.b3 = sum(dY,2);

dH2 = model.W3'*dY;
dZ2 = dH2.*(1-H2.^2);

grads.W2 = dZ2*H1' + 2*opt.weightDecay*model.W2;
grads.b2 = sum(dZ2,2);

dH1 = model.W2'*dZ2;
dZ1 = dH1.*(1-H1.^2);

grads.W1 = dZ1*X' + 2*opt.weightDecay*model.W1;
grads.b1 = sum(dZ1,2);

dX = model.W1'*dZ1;
dEbatch = dX(3:end,:) + 2*opt.embeddingPenalty*Ebatch/numel(Ebatch);

grads.E = zeros(size(model.E));
for k = 1:numel(subjects)
    grads.E(:,subjects(k)) = grads.E(:,subjects(k)) + dEbatch(:,k);
end
end

function grads = clipGradients(grads,threshold)
fields = fieldnames(grads);
normSq = 0;
for i = 1:numel(fields)
    g = grads.(fields{i});
    normSq = normSq + sum(g(:).^2);
end
globalNorm = sqrt(normSq);

if globalNorm > threshold
    factor = threshold/(globalNorm+eps);
    for i = 1:numel(fields)
        grads.(fields{i}) = grads.(fields{i})*factor;
    end
end
end

function model = adamUpdate(model,grads,lr)
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;

model.adamStep = model.adamStep+1;
t = model.adamStep;

fields = fieldnames(grads);
for i = 1:numel(fields)
    name = fields{i};
    g = grads.(name);

    model.adamM.(name) = beta1*model.adamM.(name)+(1-beta1)*g;
    model.adamV.(name) = beta2*model.adamV.(name)+(1-beta2)*(g.^2);

    mHat = model.adamM.(name)/(1-beta1^t);
    vHat = model.adamV.(name)/(1-beta2^t);

    model.(name) = model.(name)-lr*mHat./(sqrt(vHat)+epsilon);
end
end

function e = adaptEmbedding(model,cal,opt)
e = zeros(size(model.E,1),1);
m = zeros(size(e));
v = zeros(size(e));

beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;

x = cal.xs';
u = cal.us';
y = cal.ys';

for epoch = 1:opt.adaptationEpochs
    [~,gradE] = embeddingLossAndGradient(model,e,x,u,y,opt);

    if norm(gradE)>opt.gradientClip
        gradE = gradE*opt.gradientClip/(norm(gradE)+eps);
    end

    m = beta1*m+(1-beta1)*gradE;
    v = beta2*v+(1-beta2)*(gradE.^2);

    mHat = m/(1-beta1^epoch);
    vHat = v/(1-beta2^epoch);

    e = e-opt.embeddingLearningRate*mHat./(sqrt(vHat)+epsilon);
end
end

function [loss,gradE] = embeddingLossAndGradient(model,e,x,u,y,opt)
n = numel(x);
Ebatch = repmat(e,1,n);
X = [x;u;Ebatch];

H1 = tanh(model.W1*X+model.b1);
H2 = tanh(model.W2*H1+model.b2);
Yhat = model.W3*H2+model.b3;

err = Yhat-y;
loss = mean(err.^2)+opt.embeddingPenalty*mean(e.^2);

dY = 2*err/n;
dH2 = model.W3'*dY;
dZ2 = dH2.*(1-H2.^2);
dH1 = model.W2'*dZ2;
dZ1 = dH1.*(1-H1.^2);
dX = model.W1'*dZ1;

gradE = sum(dX(3:end,:),2)+2*opt.embeddingPenalty*e/numel(e);
end

function model = fitCubicScratchModel(cal)
X = [ones(numel(cal.xs),1), cal.xs(:), cal.xs(:).^2, cal.xs(:).^3, cal.us(:)];
lambda = 1e-4;
penalty = diag([0 1 1 1 1]);
model.beta = (X'*X + lambda*penalty)\(X'*cal.ys(:));
end

function ys = cubicPredictStandardised(model,xs,us)
X = [ones(numel(xs),1), xs(:), xs(:).^2, xs(:).^3, us(:)];
ys = X*model.beta;
end

function ys = sharedPredictStandardised(model,e,xs,us)
n = numel(xs);
X = [xs(:)'; us(:)'; repmat(e,1,n)];
H1 = tanh(model.W1*X+model.b1);
H2 = tanh(model.W2*H1+model.b2);
ys = (model.W3*H2+model.b3)';
end

function drift = cubicPredictDrift(model,x,u,scale)
xs = (x-scale.xMean)/scale.xStd;
us = (u-scale.uMean)/scale.uStd;
ys = cubicPredictStandardised(model,xs,us);
drift = ys*scale.yStd+scale.yMean;
end

function drift = sharedPredictDrift(model,e,x,u,scale)
xs = (x-scale.xMean)/scale.xStd;
us = (u-scale.uMean)/scale.uStd;
ys = sharedPredictStandardised(model,e,xs,us);
drift = ys*scale.yStd+scale.yMean;
end

function x = rolloutCubic(model,initialState,u,cfg,scale)
T = numel(u);
x = zeros(T,1);
x(1) = initialState;
for t = 1:T-1
    drift = cubicPredictDrift(model,x(t),u(t),scale);
    x(t+1) = x(t)+cfg.dt*drift;
    x(t+1) = max(min(x(t+1),2.3),-2.3);
end
end

function x = rolloutShared(model,e,initialState,u,cfg,scale)
T = numel(u);
x = zeros(T,1);
x(1) = initialState;
for t = 1:T-1
    drift = sharedPredictDrift(model,e,x(t),u(t),scale);
    x(t+1) = x(t)+cfg.dt*drift;
    x(t+1) = max(min(x(t+1),2.3),-2.3);
end
end

function V = recoverCubicPotential(model,xGrid,scale)
drift = cubicPredictDrift(model,xGrid(:),zeros(numel(xGrid),1),scale);
V = -cumtrapz(xGrid,drift(:)');
V = V-min(V);
end

function V = recoverSharedPotential(model,e,xGrid,scale)
drift = sharedPredictDrift(model,e,xGrid(:),zeros(numel(xGrid),1),scale);
V = -cumtrapz(xGrid,drift(:)');
V = V-min(V);
end

function V = alignPotentialOffsetOnly(V)
V = V(:)';
V = V-min(V);
end

function features = landscapeFeatures(xGrid,V)
V = V(:)';
dV = diff(V);

minIdx = find([false, dV(1:end-1)<0 & dV(2:end)>=0, false]);

if isempty(minIdx)
    [~,minIdx] = min(V);
end

[~,order] = sort(V(minIdx),'ascend');
minIdx = sort(minIdx(order(1:min(2,numel(order)))));

if numel(minIdx)>=2
    between = minIdx(1):minIdx(end);
    [~,localMax] = max(V(between));
    saddleIdx = between(localMax);
else
    [~,saddleIdx] = max(V);
end

features.minimaX = xGrid(minIdx);
features.saddleX = xGrid(saddleIdx);
features.barrier = V(saddleIdx)-mean(V(minIdx));
end

function err = matchedAttractorError(est,truth)
if isempty(est) || isempty(truth)
    err = NaN;
    return;
end
est = sort(est(:));
truth = sort(truth(:));
k = min(numel(est),numel(truth));
err = mean(abs(est(1:k)-truth(1:k))) + 0.5*abs(numel(est)-numel(truth));
end

function didTransition = basinTransition(x,saddle)
initialSide = sign(x(1)-saddle);
finalSide = sign(mean(x(max(1,end-60):end))-saddle);

if initialSide==0
    initialSide = -1;
end
if finalSide==0
    finalSide = initialSide;
end
didTransition = initialSide~=finalSide;
end

function count = countBasinTransitions(x,saddle)
side = sign(x-saddle);
side(side==0) = 1;
count = sum(abs(diff(side))>0);
end

function curve = computeTransitionCurveTrue(p,cfg,initialState)
curve = zeros(size(cfg.testAmplitudeGrid));

for k = 1:numel(cfg.testAmplitudeGrid)
    u = zeros(cfg.testSteps,1);
    stopIdx = min(cfg.testSteps, cfg.testPulseStart+cfg.testPulseDuration-1);
    u(cfg.testPulseStart:stopIdx) = cfg.testAmplitudeGrid(k);

    x = simulateSubjectDeterministicTrue(p,u,cfg,initialState);
    V = truePotential(cfg.xGrid,p);
    V = V - min(V);
    features = landscapeFeatures(cfg.xGrid, V);
    curve(k) = double(basinTransition(x,features.saddleX));
end
end

function x = simulateSubjectDeterministicTrue(p,u,cfg,initialState)
T = numel(u);
x = zeros(T,1);
x(1) = initialState;
for t = 1:T-1
    x(t+1) = x(t)+cfg.dt*trueDrift(x(t),u(t),p);
    x(t+1) = max(min(x(t+1),2.3),-2.3);
end
end

function curve = computeTransitionCurveCubic(model,cfg,scale,initialState)
curve = zeros(size(cfg.testAmplitudeGrid));

for k = 1:numel(cfg.testAmplitudeGrid)
    u = zeros(cfg.testSteps,1);
    stopIdx = min(cfg.testSteps, cfg.testPulseStart+cfg.testPulseDuration-1);
    u(cfg.testPulseStart:stopIdx) = cfg.testAmplitudeGrid(k);

    x = rolloutCubic(model,initialState,u,cfg,scale);
    V = recoverCubicPotential(model,cfg.xGrid,scale);
    features = landscapeFeatures(cfg.xGrid,V);
    curve(k) = double(basinTransition(x,features.saddleX));
end
end

function curve = computeTransitionCurveShared(model,e,cfg,scale,initialState)
curve = zeros(size(cfg.testAmplitudeGrid));

for k = 1:numel(cfg.testAmplitudeGrid)
    u = zeros(cfg.testSteps,1);
    stopIdx = min(cfg.testSteps, cfg.testPulseStart+cfg.testPulseDuration-1);
    u(cfg.testPulseStart:stopIdx) = cfg.testAmplitudeGrid(k);

    x = rolloutShared(model,e,initialState,u,cfg,scale);
    V = recoverSharedPotential(model,e,cfg.xGrid,scale);
    features = landscapeFeatures(cfg.xGrid,V);
    curve(k) = double(basinTransition(x,features.saddleX));
end
end

function initialState = initialStateFromParams(p)
initialState = -sqrt(max(p.c/p.a,0.05));
end

function X = simulateStochasticTrueEnsemble(p,u,cfg,initialState,nRep,seed)
rng(seed,'twister');
T = numel(u); X = zeros(T,nRep); X(1,:) = initialState;
for t = 1:T-1
    drift = trueDrift(X(t,:),u(t),p);
    X(t+1,:) = X(t,:) + cfg.dt*drift + p.sigma*sqrt(cfg.dt)*randn(1,nRep);
    X(t+1,:) = max(min(X(t+1,:),2.3),-2.3);
end
end

function X = simulateStochasticCubicEnsemble(model,u,cfg,scale,initialState,nRep,sigma,seed)
rng(seed,'twister');
T = numel(u); X = zeros(T,nRep); X(1,:) = initialState;
for t = 1:T-1
    drift = cubicPredictDrift(model,X(t,:)',repmat(u(t),nRep,1),scale)';
    X(t+1,:) = X(t,:) + cfg.dt*drift + sigma*sqrt(cfg.dt)*randn(1,nRep);
    X(t+1,:) = max(min(X(t+1,:),2.3),-2.3);
end
end

function X = simulateStochasticSharedEnsemble(model,e,u,cfg,scale,initialState,nRep,sigma,seed)
rng(seed,'twister');
T = numel(u); X = zeros(T,nRep); X(1,:) = initialState;
for t = 1:T-1
    drift = sharedPredictDrift(model,e,X(t,:)',repmat(u(t),nRep,1),scale)';
    X(t+1,:) = X(t,:) + cfg.dt*drift + sigma*sqrt(cfg.dt)*randn(1,nRep);
    X(t+1,:) = max(min(X(t+1,:),2.3),-2.3);
end
end

function prob = empiricalProbability(samples,edges,pseudoCount)
counts = histcounts(samples(:),edges,'Normalization','count');
prob = counts + pseudoCount;
prob = prob/sum(prob);
end

function d = klDivergence(p,q)
p = p(:)/sum(p); q = q(:)/sum(q);
d = sum(p.*log(p./q));
end

function d = jsDivergence(p,q)
p = p(:)/sum(p); q = q(:)/sum(q); m = 0.5*(p+q);
d = 0.5*sum(p.*log(p./m)) + 0.5*sum(q.*log(q./m));
end

function value = meanResponseJSDivergence(trueX,modelX,edges,stride,pseudoCount)
idx = unique([1:stride:size(trueX,1), size(trueX,1)]);
vals = zeros(numel(idx),1);
for k = 1:numel(idx)
    p = empiricalProbability(trueX(idx(k),:),edges,pseudoCount);
    q = empiricalProbability(modelX(idx(k),:),edges,pseudoCount);
    vals(k) = jsDivergence(p,q);
end
value = mean(vals);
end

function curve = computeTransitionProbabilityCurveTrue(p,cfg,initialState,nRep,seed)
curve = zeros(size(cfg.testAmplitudeGrid));
V = truePotential(cfg.xGrid,p); V = V-min(V);
features = landscapeFeatures(cfg.xGrid,V);
for k = 1:numel(cfg.testAmplitudeGrid)
    u = zeros(cfg.testSteps,1);
    stopIdx = min(cfg.testSteps,cfg.testPulseStart+cfg.testPulseDuration-1);
    u(cfg.testPulseStart:stopIdx) = cfg.testAmplitudeGrid(k);
    X = simulateStochasticTrueEnsemble(p,u,cfg,initialState,nRep,seed+k);
    curve(k) = ensembleTransitionProbability(X,features.saddleX);
end
end

function curve = computeTransitionProbabilityCurveCubic(model,cfg,scale,initialState,sigma,nRep,seed)
curve = zeros(size(cfg.testAmplitudeGrid));
V = recoverCubicPotential(model,cfg.xGrid,scale);
features = landscapeFeatures(cfg.xGrid,V);
for k = 1:numel(cfg.testAmplitudeGrid)
    u = zeros(cfg.testSteps,1);
    stopIdx = min(cfg.testSteps,cfg.testPulseStart+cfg.testPulseDuration-1);
    u(cfg.testPulseStart:stopIdx) = cfg.testAmplitudeGrid(k);
    X = simulateStochasticCubicEnsemble(model,u,cfg,scale,initialState,nRep,sigma,seed+k);
    curve(k) = ensembleTransitionProbability(X,features.saddleX);
end
end

function curve = computeTransitionProbabilityCurveShared(model,e,cfg,scale,initialState,sigma,nRep,seed)
curve = zeros(size(cfg.testAmplitudeGrid));
V = recoverSharedPotential(model,e,cfg.xGrid,scale);
features = landscapeFeatures(cfg.xGrid,V);
for k = 1:numel(cfg.testAmplitudeGrid)
    u = zeros(cfg.testSteps,1);
    stopIdx = min(cfg.testSteps,cfg.testPulseStart+cfg.testPulseDuration-1);
    u(cfg.testPulseStart:stopIdx) = cfg.testAmplitudeGrid(k);
    X = simulateStochasticSharedEnsemble(model,e,u,cfg,scale,initialState,nRep,sigma,seed+k);
    curve(k) = ensembleTransitionProbability(X,features.saddleX);
end
end

function p = ensembleTransitionProbability(X,saddle)
initialSide = sign(X(1,:)-saddle); initialSide(initialSide==0) = -1;
finalSide = sign(mean(X(max(1,end-60):end,:),1)-saddle);
finalSide(finalSide==0) = initialSide(finalSide==0);
p = mean(initialSide~=finalSide);
end

function summary = summariseResults(results,cfg)
metricNames = {'passiveRMSE','perturbRMSE','landscapeRMSE', ...
    'barrierError','attractorError','transitionCorrect', ...
    'amplitudeCurveRMSE','flowRMSE','invariantKL','responseJS', ...
    'transitionProbabilityRMSE'};

for i = 1:numel(metricNames)
    name = metricNames{i};
    values = results.(name);
    summary.(name).mean = squeeze(mean(values,1,'omitnan'));
    summary.(name).sem = squeeze(std(values,0,1,'omitnan')) / sqrt(cfg.testSubjects);
end
end

function makeMainFigure(cfg,trainParams,summary,example,plt)
fig = figure('Color','w','Units','centimeters','Position',[2 2 18.3 20.5]);
tl = tiledlayout(fig,3,2,'TileSpacing','compact','Padding','compact');

% A: population family
ax = nexttile(tl,1); hold(ax,'on');
idx = round(linspace(1,numel(trainParams),7));
for k = idx
    V = truePotential(cfg.xGrid,trainParams(k)); V = V-min(V);
    plot(ax,cfg.xGrid,V,'Color',[0.55 0.55 0.55],'LineWidth',1.0);
end
xlabel(ax,'State, $x$','Interpreter','latex');
ylabel(ax,'Potential, $V(x)$','Interpreter','latex');
title(ax,'A  Population dynamical family','Interpreter','latex'); formatAxes(ax);

% B: calibration episode with legend
ax = nexttile(tl,2); hold(ax,'on');
timeCal = (0:numel(example.calibrationX)-1)*cfg.dt;
yyaxis(ax,'left');
hState = plot(ax,timeCal,example.calibrationX,'Color',plt.groundTruth,'LineWidth',1.6);
hSamples = scatter(ax,example.selectedIdx*cfg.dt,example.selectedX,18,'filled', ...
    'MarkerFaceColor',plt.perturb,'MarkerEdgeColor','none');
ylabel(ax,'State, $x$','Interpreter','latex');
yyaxis(ax,'right');
hInput = stairs(ax,timeCal,example.calibrationU,'Color',[0.45 0.45 0.45],'LineWidth',1.1);
ylabel(ax,'Input, $u$','Interpreter','latex');
xlabel(ax,'Time, $t$','Interpreter','latex');
title(ax,sprintf('B  Few-shot calibration (%d samples)',example.nCal),'Interpreter','latex');
leg = legend(ax,[hState,hSamples,hInput],{'State trajectory','Selected calibration samples','Input drive'}, ...
    'Interpreter','latex','Location','northwest','Box','off','FontSize',8.5);
set(leg,'AutoUpdate','off');
formatAxes(ax);

% C: controlled vector fields at representative inputs
ax = nexttile(tl,3); hold(ax,'on');
inputToShow = [0 cfg.testPulseAmplitude];
lineStyles = {'-','--'};
for q = 1:numel(inputToShow)
    u0 = inputToShow(q);
    trueF = trueDrift(cfg.xGrid,u0,example.params);
    [~,uCol] = min(abs(cfg.flowInputGrid-u0));
    passiveF = example.flowPredictions{2}(:,uCol);
    pertF = example.flowPredictions{3}(:,uCol);
    plot(ax,cfg.xGrid,trueF,lineStyles{q},'Color',plt.groundTruth,'LineWidth',2.3);
    plot(ax,cfg.xGrid,passiveF,lineStyles{q},'Color',plt.passive,'LineWidth',2.2);
    plot(ax,cfg.xGrid,pertF,lineStyles{q},'Color',plt.perturb,'LineWidth',2.3);
end
yline(ax,0,'Color',[0.75 0.75 0.75],'LineWidth',0.9);
xlabel(ax,'State, $x$','Interpreter','latex');
ylabel(ax,'Flow, $f(x,u)$','Interpreter','latex');
title(ax,'C  Controlled vector-field reconstruction','Interpreter','latex');
legend(ax,{ ...
    'Truth, $u=0$','Passive, $u=0$','Perturbational, $u=0$', ...
    'Truth, held-out $u$','Passive, held-out $u$','Perturbational, held-out $u$'}, ...
    'Interpreter','latex','Location','best','Box','off','FontSize',8.0);
formatAxes(ax);

% D: estimated energy landscapes
ax = nexttile(tl,4); hold(ax,'on');
plot(ax,cfg.xGrid,example.Vtrue,'-','Color',plt.groundTruth,'LineWidth',2.4);
plot(ax,cfg.xGrid,example.Vset{1},'--','Color',plt.scratch,'LineWidth',1.9);
plot(ax,cfg.xGrid,example.Vset{2},':','Color',plt.passive,'LineWidth',2.2);
plot(ax,cfg.xGrid,example.Vset{3},'-','Color',plt.perturb,'LineWidth',2.3);
xlabel(ax,'State, $x$','Interpreter','latex');
ylabel(ax,'Energy, $V(x)$','Interpreter','latex');
title(ax,'D  Estimated energy landscapes','Interpreter','latex');
legend(ax,{'Ground truth','Scratch','Passive pretraining','Perturbational pretraining'}, ...
    'Interpreter','latex','Location','best','Box','off','FontSize',8.5);
formatAxes(ax);

% E: held-out perturbation response distributions
ax = nexttile(tl,5); hold(ax,'on');
time = (0:cfg.testSteps-1)*cfg.dt;
[muT,loT,hiT] = ensembleBand(example.trueResponseEnsemble);
[muPa,loPa,hiPa] = ensembleBand(example.modelResponseEnsembles{2});
[muPe,loPe,hiPe] = ensembleBand(example.modelResponseEnsembles{3});
fillBand(ax,time,loT,hiT,[0.75 0.75 0.75],0.32);
fillBand(ax,time,loPa,hiPa,plt.passive,0.12);
fillBand(ax,time,loPe,hiPe,plt.perturb,0.12);
plot(ax,time,muT,'Color',plt.groundTruth,'LineWidth',2.3);
plot(ax,time,muPa,'Color',plt.passive,'LineWidth',2.1);
plot(ax,time,muPe,'Color',plt.perturb,'LineWidth',2.3);
xline(ax,cfg.testPulseStart*cfg.dt,':','Color',[0.45 0.45 0.45]);
xline(ax,(cfg.testPulseStart+cfg.testPulseDuration)*cfg.dt,':','Color',[0.45 0.45 0.45]);
xlabel(ax,'Time, $t$','Interpreter','latex');
ylabel(ax,'State, $x$','Interpreter','latex');
title(ax,'E  Held-out perturbational response','Interpreter','latex');
legend(ax,{ ...
    'Truth: 90\% interval','Passive: 90\% interval','Perturbational: 90\% interval', ...
    'Truth: ensemble mean','Passive: ensemble mean','Perturbational: ensemble mean'}, ...
    'Interpreter','latex','Location','best','Box','off','FontSize',7.8);
formatAxes(ax);

% F: stochastic dose-transition curve
ax = nexttile(tl,6); hold(ax,'on');
plot(ax,cfg.testAmplitudeGrid,example.trueProbCurve,'-','Color',plt.groundTruth,'LineWidth',2.4);
plot(ax,cfg.testAmplitudeGrid,example.modelProbCurves{1},'--o','Color',plt.scratch,'LineWidth',1.7,'MarkerSize',4.2);
plot(ax,cfg.testAmplitudeGrid,example.modelProbCurves{2},':s','Color',plt.passive,'LineWidth',2.0,'MarkerSize',4.2);
plot(ax,cfg.testAmplitudeGrid,example.modelProbCurves{3},'-d','Color',plt.perturb,'LineWidth',2.3,'MarkerSize',4.2);
xlabel(ax,'Pulse amplitude, $u$','Interpreter','latex');
ylabel(ax,'Transition probability','Interpreter','latex');
ylim(ax,[-0.03 1.03]);
title(ax,'F  Intervention outcome','Interpreter','latex');
legend(ax,{'Ground truth','Scratch','Passive pretraining','Perturbational pretraining'}, ...
    'Interpreter','latex','Location','southeast','Box','off','FontSize',8.5);
formatAxes(ax);

sgtitle(tl,'Perturbational foundation-model transfer: trajectory, structure, and response', ...
    'FontWeight','bold','FontSize',14,'Interpreter','latex');
exportPublicationFigure(fig,cfg.outputDir,'Figure_3_mechanistic_validation');
end

function makeSupplementFigure(cfg,summary,example,plt)
fig = figure('Color','w','Units','centimeters','Position',[2 2 18.3 18.5]);
tl = tiledlayout(fig,3,2,'TileSpacing','compact','Padding','compact');
metrics = { ...
    'passiveRMSE','Time-series accuracy','Passive rollout RMSE',false; ...
    'flowRMSE','Dynamical structure','Controlled flow RMSE',false; ...
    'invariantKL','Dynamical structure','Invariant-distribution KL',false; ...
    'responseJS','Perturbational response','Response-distribution JS',false; ...
    'transitionProbabilityRMSE','Perturbational response','Dose--transition RMSE',false; ...
    'attractorError','Geometric interpretability','Attractor-location error',false};
for k = 1:size(metrics,1)
    ax = nexttile(tl,k); hold(ax,'on');
    plotLearningCurve(ax,cfg,summary.(metrics{k,1}),plt);
    xlabel(ax,'Calibration samples','Interpreter','latex'); ylabel(ax,metrics{k,3},'Interpreter','latex');
    title(ax,sprintf('%c  %s',char('A'+k-1),metrics{k,2}),'Interpreter','latex');
    formatAxes(ax);
    if k==1
        legend(ax,{'Scratch','Passive pretraining','Perturbational pretraining'}, ...
            'Location','best','Box','off','FontSize',8.5);
    end
end
sgtitle(tl,'Few-shot transfer across complementary validation criteria', ...
    'FontWeight','bold','FontSize',14);
exportPublicationFigure(fig,cfg.outputDir,'Figure_4_few_shot_metrics');
end

function makeCoverageFigure(cfg,passiveData,perturbData,results,plt)
fig = figure('Color','w','Units','centimeters','Position',[2 2 18.3 8.8]);
tl = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');

ax = nexttile(tl,1); hold(ax,'on');
histogram(ax,passiveData.x,cfg.densityEdges,'Normalization','probability', ...
    'FaceColor',[0.70 0.70 0.70],'EdgeColor','none','FaceAlpha',0.85);
histogram(ax,perturbData.x,cfg.densityEdges,'Normalization','probability', ...
    'DisplayStyle','stairs','EdgeColor',plt.perturb,'LineWidth',1.8);
xlabel(ax,'State, $x$','Interpreter','latex'); ylabel(ax,'Probability','Interpreter','latex');
title(ax,'A  State-space coverage','Interpreter','latex');
legend(ax,{'Passive','Perturbational'},'Box','off','Location','best'); formatAxes(ax);

ax = nexttile(tl,2);
histogram(ax,perturbData.u,60,'Normalization','probability', ...
    'FaceColor',[0.55 0.55 0.55],'EdgeColor','none');
xlabel(ax,'Input, $u$','Interpreter','latex'); ylabel(ax,'Probability','Interpreter','latex');
title(ax,'B  Input-space coverage','Interpreter','latex'); formatAxes(ax);

ax = nexttile(tl,3); coverage = results.coverageStats;
means = mean(coverage(:,1:4),1,'omitnan');
sems = std(coverage(:,1:4),0,1,'omitnan')/sqrt(size(coverage,1));
b = bar(ax,1:4,means,'FaceColor',[0.65 0.65 0.65],'EdgeColor','none'); %#ok<NASGU>
hold(ax,'on'); errorbar(ax,1:4,means,sems,'k.','LineWidth',1.1);
set(ax,'XTick',1:4,'XTickLabel',{'Left','Boundary','Right','Input on'});
ylabel(ax,'Fraction of calibration samples','Interpreter','latex'); title(ax,'C  New-subject excitation','Interpreter','latex');
formatAxes(ax);

sgtitle(tl,'Coverage and excitation diagnostics','FontWeight','bold','FontSize',14);
exportPublicationFigure(fig,cfg.outputDir,'Figure_2_coverage');
end

function plotLearningCurve(ax,cfg,metric,plt)
% Positive-safe asymmetric error bars permit clean log--log visualization.
styles = {'--o',':s','-d'};
colors = {plt.scratch,plt.passive,plt.perturb};
widths = [2.0 2.2 2.5];
faces = {'w','w',plt.perturb};

for m = 1:3
    y = max(metric.mean(:,m),1e-8);
    sem = max(metric.sem(:,m),0);
    lowerError = min(sem,0.80*y);  % prevents non-positive lower limits on log axes
    upperError = sem;
    errorbar(ax,cfg.calibrationSizes,y,lowerError,upperError,styles{m}, ...
        'Color',colors{m},'LineWidth',widths(m),'MarkerSize',5.5, ...
        'MarkerFaceColor',faces{m},'CapSize',7);
end

set(ax,'XScale','log','YScale',cfg.learningCurveYScale, ...
    'XTick',cfg.calibrationSizes,'XTickLabel',string(cfg.calibrationSizes));
xlim(ax,[min(cfg.calibrationSizes)*0.85 max(cfg.calibrationSizes)*1.15]);
end

function exportPublicationFigure(fig,outputDir,baseName)
% Export vector and raster formats, with PRINT as the TIFF fallback.

set(fig,'Renderer','painters');

pdfFile = fullfile(outputDir,[baseName '.pdf']);
pngFile = fullfile(outputDir,[baseName '.png']);
tifFile = fullfile(outputDir,[baseName '.tif']);
figFile = fullfile(outputDir,[baseName '.fig']);

exportgraphics(fig,pdfFile,'ContentType','vector');
exportgraphics(fig,pngFile,'Resolution',600);

try
    exportgraphics(fig,tifFile,'Resolution',600);
catch ME
    warning('exportgraphics TIFF export failed (%s). Falling back to PRINT.',ME.message);
    print(fig,tifFile,'-dtiff','-r600');
end

savefig(fig,figFile);
end

function fillBand(ax,x,lo,hi,color,alphaValue)
fill(ax,[x(:);flipud(x(:))],[lo(:);flipud(hi(:))],color, ...
    'FaceAlpha',alphaValue,'EdgeColor','none');
end

function [mu,lo,hi] = ensembleBand(X)
mu = mean(X,2,'omitnan');
lo = prctile(X,5,2); hi = prctile(X,95,2);
end

function formatAxes(ax)
set(ax,'Box','off','FontName','Arial','FontSize',11.0, ...
    'FontWeight','normal','LineWidth',1.25,'TickDir','out', ...
    'TickLength',[0.018 0.018],'Layer','top');
ax.XLabel.FontSize = 12.0;
ax.YLabel.FontSize = 12.0;
ax.Title.FontSize = 12.0;
ax.Title.FontWeight = 'bold';
grid(ax,'off');
end

function writeSummaryTable(cfg,summary)
modelNames = ["CubicScratch";"PassivePretraining";"PerturbationalPretraining"];

nC = numel(cfg.calibrationSizes);
rows = nC*3;

CalibrationSamples = zeros(rows,1);
Model = strings(rows,1);
PassiveRMSE = zeros(rows,1);
PerturbRMSE = zeros(rows,1);
LandscapeRMSE = zeros(rows,1);
BarrierError = zeros(rows,1);
AttractorError = zeros(rows,1);
TransitionAccuracy = zeros(rows,1);
AmplitudeCurveRMSE = zeros(rows,1);
FlowRMSE = zeros(rows,1);
InvariantKL = zeros(rows,1);
ResponseJS = zeros(rows,1);
TransitionProbabilityRMSE = zeros(rows,1);

r = 0;
for ci = 1:nC
    for m = 1:3
        r = r+1;
        CalibrationSamples(r) = cfg.calibrationSizes(ci);
        Model(r) = modelNames(m);
        PassiveRMSE(r) = summary.passiveRMSE.mean(ci,m);
        PerturbRMSE(r) = summary.perturbRMSE.mean(ci,m);
        LandscapeRMSE(r) = summary.landscapeRMSE.mean(ci,m);
        BarrierError(r) = summary.barrierError.mean(ci,m);
        AttractorError(r) = summary.attractorError.mean(ci,m);
        TransitionAccuracy(r) = summary.transitionCorrect.mean(ci,m);
        AmplitudeCurveRMSE(r) = summary.amplitudeCurveRMSE.mean(ci,m);
        FlowRMSE(r) = summary.flowRMSE.mean(ci,m);
        InvariantKL(r) = summary.invariantKL.mean(ci,m);
        ResponseJS(r) = summary.responseJS.mean(ci,m);
        TransitionProbabilityRMSE(r) = summary.transitionProbabilityRMSE.mean(ci,m);
    end
end

T = table(CalibrationSamples,Model,PassiveRMSE,PerturbRMSE, ...
    LandscapeRMSE,BarrierError,AttractorError,TransitionAccuracy,AmplitudeCurveRMSE, ...
    FlowRMSE,InvariantKL,ResponseJS,TransitionProbabilityRMSE);

writetable(T,fullfile(cfg.outputDir,'summary_metrics_visual.csv'));
end
