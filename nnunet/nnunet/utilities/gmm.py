from batchgenerators.utilities.file_and_folder_operations import load_pickle
import numpy as np
from skimage.measure import label
import nibabel as nib
import pandas as pd
import copy
import random
import math
import glob
import os
import pdb

# GMM for menigioma project
# predict using nnUNet_predict with argument--save_npz to get softmax prediction in .npz file

def get_highest_sum_probability_region(mask3d, prob3d, n_candidate=10):
    if np.sum(mask3d) == 0:
        return mask3d
    """ return region with highest sum of probability in prob3d"""
    labels = label(mask3d)  # label each connected region with index from 0 - n of connected region found
    n_connected_region = np.bincount(labels.flat)  # number of pixel for each connected region
    biggest_regions_index = (-n_connected_region).argsort()[1:n_candidate+1]  # get n biggest regions index without BG

    #biggest_regions = np.array([])
    max_sum = 0
    for ind in biggest_regions_index:
        current_region = prob3d * (labels == ind)
        # COUNT SUM
        prob_sum = np.sum(current_region)
        if prob_sum > max_sum:
            max_sum = prob_sum
            max_ind = ind
        '''
        if biggest_regions.size == 0:
            biggest_regions = labels == ind
        else:
            pdb.set_trace()
            biggest_regions += labels == ind
        '''
    #pdb.set_trace()
    return labels == max_ind

class GMMOverlapedData:
    MeanCluster = 0
    SD = 0
    Prior = 0
    def __lt__(self, other):
         return self.MeanCluster < other.MeanCluster

class KMeans:
    """KMeans"""
    m_dimNum = 1
    m_clusterNum = 1
	#double** m_means;
    #int m_initMode;
    InitRandom = 0 
    InitManual = 1
    InitUniform = 2

    # parameterized constructor
    def __init__(self, dimNum, clusterNum):
        self.m_dimNum = dimNum
        self.m_clusterNum = clusterNum

        self.m_means = np.zeros((self.m_clusterNum, self.m_dimNum), dtype=float)
        self.m_initMode = self.InitUniform
        #self.InitRandom
        self.m_maxIterNum = 100
        self.m_endError = 0.001

    def GetMean(self,i):
        return self.m_means[i]

    # Euclidean distance
    def CalcDistance(self, x, u, dimNum):
        temp = 0.0
        for d in range(dimNum):
            temp += (x[d] - u[d]) * (x[d] - u[d])
        return math.sqrt(temp)

    def GetLabel(self, sample, label):
        dist = -1
        for i in range(self.m_clusterNum): # 모든 클러스터에 대해 >
            temp = self.CalcDistance(sample, self.m_means[i], self.m_dimNum)
            #각 평균값과 가장 가까운 클러스터를 찾는다.
            if temp < dist or dist == -1:
                dist = temp
                label = i #가장 가까운 label을 저장하고, >
        return dist, label #cost인 유클리디안-거리는 return으로 넘겨준다.
    
    def Cluster(self, data, N, Label):
        #double *data, int N, int *Label
        size = N
        assert(size >= self.m_clusterNum)

        #Initialize model
        self.Init(data, N) #클러스터별 초기 평균값을 정해준다.

        #Recursion
        x = np.zeros((self.m_dimNum), dtype=float)
        label = -1 #Class index
        iterNum = 0.0
        lastCost = 0.0
        currCost = 0.0
        unchanged = 0
        counts = np.zeros((self.m_clusterNum), dtype=int)
        #New model for reestimation
        next_means = np.zeros((self.m_clusterNum, self.m_dimNum), dtype=float)
        loop = True
        while loop == True:
            counts.fill(0)
            for i in range(self.m_clusterNum):
                next_means[i].fill(0) #새로운 평균값을 저장
            lastCost = currCost
            currCost = 0
            #Classification
            for i in range(size):
                for j in range(self.m_dimNum):
                    x[j] = data[i * self.m_dimNum + j] # 입력샘플 하나를 뽑아서
                cost, label = self.GetLabel(x, label) #해당 샘플의 cost와 label을 구하고
                currCost += cost
                #해당 샘플의 cost와 label을 구하고
                # currCost는 모든 입력샘플의 유클리디안-거리의 합 (모든 입력샘플은 각각 하나의 클러스터에 포함된다.)
                
                # 해당 label의 카운터 증가, 새로운 평균값을 구할 때 각 클러스터(label)당 입력샘플의 개수로 사용한다.
                counts[label] += 1
                for d in range(self.m_dimNum):
                    next_means[label][d] += x[d]
                    #다음 Re-estimation에서 평균값을 다시 구하기 위해, next_means에 label이 구해진 입력샘플을 모두 더한다.
            
            #입력샘플의 개수로 나눠서, cost(유클리디안-거리)의 평균을 구한다. 의미가 있나?
            currCost /= size
            #Reestimation
            for i in range(self.m_clusterNum):
                if counts[i] > 0:
                    for d in range(self.m_dimNum):
                        next_means[i][d] /= counts[i] # 새로운 평균값을 구한다.
                    self.m_means[i] = copy.deepcopy(next_means[i])
                    #평균값을 업데이트한다.
            #Terminal conditions
            iterNum += 1
            #cost의 변화가 거의 없을 경우(아마 아주 적은 갯수의 샘플이동), 카운팅을 해주고 >
            if math.fabs(lastCost - currCost) < self.m_endError * lastCost:
                unchanged += 1
            if iterNum >= self.m_maxIterNum or unchanged >= 3: #3회 이상이면 종료
                loop = False
        # Output the label file
        for i in range(size): #모든 입력 샘플의 >
            for j in range(self.m_dimNum):
                x[j] = data[i * self.m_dimNum + j]
            cost, label = self.GetLabel(x, label) #label을 구해서, 결과로 확정하고 > 
            Label[i] = label  #저장한다.


    def Init(self, data, N):# N: 입력샘플 개수
        size = N
        if self.m_initMode == self.InitRandom: #랜덤한 위치에서 입력샘플을 뽑아서 클러스터의 초기 평균값으로 지정.
            inteval = int(size / self.m_clusterNum)
            sample = np.zeros((self.m_dimNum), dtype=float)

            # Seed the random-number generator with current time
            #srand((unsigned)time(NULL));
            for i in range(self.m_clusterNum):
                select = int(inteval * i + (inteval - 1) * random.uniform(0, 1))
                for j in range(self.m_dimNum):
                    sample[j] = data[select * self.m_dimNum + j]
                self.m_means[i] = copy.deepcopy(sample)
                #memcpy(m_means[i], sample, sizeof(double) * m_dimNum);
        elif self.m_initMode == self.InitUniform:
            sample = np.zeros((self.m_dimNum), dtype=int)
            for i in range(self.m_clusterNum):
                select = int(i * size / self.m_clusterNum) #클러스터 개수만큼 나눠서 >
                for j in range(self.m_dimNum):
                    # 나눠진 그룹의 시작이 되는 원소(입력샘플)를 >
                    sample[j] = data[select * self.m_dimNum + j]
                self.m_means[i] = copy.deepcopy(sample)
        #elif self.m_initMode == self.InitManual:
        # Do nothing

class GMM:
    """GMM"""
    m_dimNum = 0
    m_mixNum = 0
    m_maxIterNum = 100
    m_endError = 0.001

    # parameterized constructor
    def __init__(self, dimNum, mixNum):
        self.m_dimNum = dimNum
        self.m_mixNum = mixNum

        self.m_maxIterNum = 100
        self.m_endError = 0.001

        self.countHuIndex = 0;
        self.countWholeHuValue = 0;
        
        self.m_priors = np.zeros((self.m_mixNum), dtype=float)
        self.m_means = np.zeros((self.m_mixNum, self.m_dimNum), dtype=float)
        self.m_vars = np.zeros((self.m_mixNum, self.m_dimNum), dtype=float)
        self.m_minVars = np.zeros((self.m_mixNum), dtype=float)

        for i in range(self.m_mixNum):
            self.m_priors[i] = 1.0 / self.m_mixNum
            for d in range(self.m_dimNum):
                self.m_means[i][d] = 0
                self.m_vars[i][d] = 1

    def Prior(self,i):
        return self.m_priors[i]

    def Mean(self,i):
        return self.m_means[i]
        
    def Variance(self,i):
        return self.m_vars[i]

    def Init(self, data, N):
        MIN_VAR = 1E-10
        kmeans = KMeans(self.m_dimNum, self.m_mixNum)
        #kmeans.SetInitMode(KMeans.InitUniform)
        '''
        int *Label;
        #입력샘플 각각에 클러스터링 레이블 부여
        Label = new int[N]
        #k-means를 통해서 모든 입력샘플에 대해 클러스터 Label를 구한다.
        '''
        Label = np.zeros((N), dtype=int)
        kmeans.Cluster(data, N, Label)
        #각 분포의 입력샘플 개수저장할 듯 // m_mixNum: 구하고자하는 정규분포 개수
        counts = np.zeros((self.m_mixNum), dtype=int)
        # 각 차원의 전체 평균? // Overall mean of training data
        overMeans = np.zeros((self.m_dimNum), dtype=float)
        for i in range(self.m_mixNum):
            counts[i] = 0
            #해당 분포의 확률
            self.m_priors[i] = 0
            #kmean로 얻은 평균값을 초기값으로 사용
            # memcpy(self.m_means[i], kmeans->GetMean(i), sizeof(double) * m_dimNum)
            self.m_vars.fill(0)
        #overMeans.fill(0)
        self.m_minVars.fill(0)
        size = N
        #입력샘플 하나를 뽑아내기 위한 변수
        x = np.zeros((self.m_dimNum), dtype=float)
        label = -1

        for i in range(size):#모든 입력샘플에 대해
            for j in range(self.m_dimNum):
                x[j] = data[i * self.m_dimNum + j] #입력 샘플과 > 
            label = Label[i] #입력샘플의 k-means로 구한 Label
            
            #Count each Gaussian
            counts[label] += 1
            m = kmeans.GetMean(label)
            for d in range(self.m_dimNum):
                #입력샘플에 대해 해당 입력샘플이 포함된 분포의 평균값과 분산(편차의 제곱)을 구함
                self.m_vars[label][d] += (x[d] - m[d]) * (x[d] - m[d])
                
            #Count the overall mean and variance.
            for d in range(self.m_dimNum):
                overMeans[d] += x[d] #각 차원마다 입력을 저장
                self.m_minVars[d] += x[d] * x[d] #0을 기준으로 한다면 이게 분산이 맞는데.. 아래서 평균을 빼주기 때문에 상관없는듯
        
        # 분산 구하는 법
        # 평균값의 선형성으로부터 다음과 같은 식을 얻을 수 있다. >> 이 내용을 참고
        # E[X^2] - (E[X])^2
        # https://ko.wikipedia.org/wiki/%EB%B6%84%EC%82%B0

        # Compute the overall variance (* 0.01) as the minimum variance.
        #for d in range()
        for d in range(self.m_dimNum):
            overMeans[d] /= size #각 차원마다 평균을 구함
            #각 차원의 전체 분산의 1퍼센트 값?
            self.m_minVars[d] = max(MIN_VAR, 0.01 * (self.m_minVars[d] / size - overMeans[d] * overMeans[d]))
            # E[] : 평균을 표현
            # m_minVars[d] / size = E[X^2]																							
            # overMeans[d] * overMeans[d] = (E[X])^2
        
        # Initialize each Gaussian.
        for i in range(self.m_mixNum):
            #분포가 차지하는 입력샘플의 비중. 즉, 입력샘플들이 분포에 포함될 확률.
            self.m_priors[i] = 1.0 * counts[i] / size
            if self.m_priors[i] > 0: #k-means로 클러스터가 구성이 됐다면, 이 값은 존재
                for d in range(self.m_dimNum):
                    self.m_vars[i][d] = self.m_vars[i][d] / counts[i]
                    # A minimum variance for each dimension is required.
                    if self.m_vars[i][d] < self.m_minVars[d]:
                        self.m_vars[i][d] = self.m_minVars[d]
            else:
                self.m_vars[i] = copy.deepcopy(self.m_minVars[0])
                #memcpy(m_vars[i], m_minVars, sizeof(double) * m_dimNum);

    def GetProbabilitySample(self, sample):
        p = 0
        for i in range(self.m_mixNum):
            p += self.m_priors[i] * self.GetProbability(sample, i)
        return p

    def GetProbability(self, x, j):
        p = 1
        for d in range(self.m_dimNum):
            divisor = math.sqrt(2 * 3.14159 * self.m_vars[j][d])
            if divisor == 0.0:
                divisor += 1e-10
            p *= 1 / divisor
            p *= math.exp(-0.5 * (x[d] - self.m_means[j][d]) * (x[d] - self.m_means[j][d]) / self.m_vars[j][d])
        return p

    def Train(self, data, N):
        self.Init(data, N)
        size = N

        # Reestimation
        loop = True
        iterNum = 0
        lastL = 0.0
        currL = 0.0
        unchanged = 0
        # Sample data
        x = np.zeros((self.m_dimNum), dtype=float)
        next_priors = np.zeros((self.m_mixNum), dtype=float)
        next_vars = np.zeros((self.m_mixNum, self.m_dimNum), dtype=float)
        next_means = np.zeros((self.m_mixNum, self.m_dimNum), dtype=float)
        #pdb.set_trace()
        while loop == True:
            # Clear buffer for reestimation
            next_priors.fill(0)
            next_vars.fill(0)
            next_means.fill(0)
            
            lastL = currL
            currL = 0
            #Predict
            for k in range(size):
                for j in range(self.m_dimNum):
                    x[j] = data[k * self.m_dimNum + j]
                p = self.GetProbabilitySample(x);
                for j in range(self.m_mixNum):
                    pj = self.GetProbability(x, j) * self.m_priors[j] / p
                    next_priors[j] += pj
                    for d in range(self.m_dimNum):
                        next_means[j][d] += pj * x[d]
                        next_vars[j][d] += pj * x[d] * x[d]
                #currL += (p > 1E-20) ? log10(p) : -20
                if p > 1E-20:
                    currL += math.log10(p)
                else:
                    currL += -20
            currL /= size
            #Reestimation: generate new priors, means and variances.
            for j in range(self.m_mixNum):
                self.m_priors[j] = next_priors[j] / size
                if (self.m_priors[j] > 0):
                    for d in range(self.m_dimNum):
                        self.m_means[j][d] = next_means[j][d] / next_priors[j]
                        self.m_vars[j][d] = next_vars[j][d] / next_priors[j] - self.m_means[j][d] * self.m_means[j][d]
                        if (self.m_vars[j][d] < self.m_minVars[d]):
                            self.m_vars[j][d] = self.m_minVars[d]
            # Terminal conditions
            iterNum += 1
            if math.fabs(currL - lastL) < self.m_endError * math.fabs(lastL):
                unchanged += 1
            if iterNum >= self.m_maxIterNum or unchanged >= 3:
                loop = False
            #--- Debug ---
            #cout << "Iter: " << iterNum << ", Average Log-Probability: " << currL << endl;
        
    def calculate(self):
        self.answer = self.first + self.second

def getGMMOverlapValue(maskVolume, voxelData, num):
    #width, height, cnt, 
    #_maskBitValue -> selected mask (ROI) index
    '''
    std::vector<short> maskedVoxel;
    for (int z = 0; z < cnt; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = z*width*height + y*width + x;
                unsigned char maskValue = maskVolume[index];
                short voxelValue = voxelData[index];
                if (maskValue & _maskBitValue) {
                    maskedVoxel.push_back(voxelValue); '''
    maskedVoxel = voxelData.flatten()
    if maskedVoxel.max() >= 0.003:
        maskedVoxel = np.delete(maskedVoxel, np.argwhere(maskedVoxel <= 0.003))
    else:
        maskedVoxel = np.delete(maskedVoxel, np.argwhere(maskedVoxel <= 0.0))
    maskVoxelSize = maskedVoxel.shape[0]
    #n = 0
    #while (!maskedVoxel.empty()) {
    #dataSet[n] = voxelValue    }
    dim = 1     
    cluster = num
    gmm = GMM(dim, cluster)
    gmm.Train(maskedVoxel, maskVoxelSize)
    #gmm.Train(dataset, maskVoxelSize)
    GMMDatas = []
    for i in range(cluster):
        #for (int i = 0; i < cluster; ++i)    
        data = GMMOverlapedData()
        data.MeanCluster = gmm.Mean(i)[0]
        data.SD = math.sqrt((gmm.Variance(i))[0])
        data.Prior = gmm.Prior(i)
        GMMDatas.append(data)
    GMMDatas.sort()
    #pdb.set_trace()
    '''
    std::sort(GMMDatas.begin(), GMMDatas.end(),
        [](const GMMOverlapedData& lhs, const GMMOverlapedData& rhs) {
        return lhs.MeanCluster < rhs.MeanCluster
    })
    '''
    return GMMDatas

target_dir = './inference_modelgmm_999/'

npzs = sorted(glob.glob(os.path.join(target_dir, '*.npz')))
pkls = sorted(glob.glob(os.path.join(target_dir, '*.pkl')))
niigzs = sorted(glob.glob(os.path.join(target_dir, '*.nii.gz')))

stats = []

for npz, pkl, niigz in zip(npzs, pkls, niigzs):
    
    softmax = np.load(npz)['softmax'][1:]
    # softmax = np.load('./inference_modelgmm_999/324_01.npz')['softmax'][1]
    # softmax['softmax'].shape

    #START GMM
    is_gmm = True
    if is_gmm:
        #pdb.set_trace()
        #if (m_vecAIresult[i][j] > 0)
        #	vecAIMaskData[j] |= maskBit;
        vecAIMaskData = softmax > 0.00392156
        num = 2
        #std::vector<GMMOverlapedData> GMMDatas
        #getGMMOverlapValue(&vecAIMaskData[0], &vecAIProbabilityData[0], cx, cy, cz, maskBit, num, GMMDatas)
        try:
            GMMDatas = getGMMOverlapValue(vecAIMaskData, softmax, num)
        except:
            print('############## FAILED ', npz)
            continue
        #gmmMid = 255.0 / 2.0
        gmmMid = 0.5
        if len(GMMDatas) >= num:
            sum = GMMDatas[num - 1].MeanCluster + GMMDatas[num - 2].MeanCluster
            gmmMid = sum / 2.0
        #nOutsetVal = (int)round(gmmMid);
        #pdb.set_trace()
        thresholds = gmmMid
    #END
    #'''
    output = softmax >= thresholds

    #'''
    #START region with highest sum of probability
    for class_output in range(0, 1):
        if np.sum(output[class_output]) > 0:
            output[class_output] = get_highest_sum_probability_region(output[class_output], \
                softmax[class_output], n_candidate=10)
            #pdb.set_trace()
    #END
    output = output[0]

    properties_dict = load_pickle(pkl)
    # properties_dict = load_pickle('./inference_modelgmm_999/324_01.pkl')
    shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')
    bbox = properties_dict.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + output.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = output
    else:
        seg_old_size = output

    nii_input = nib.load(niigz)
    # nii_input = nib.load('./inference_modelgmm_999/324_01.nii.gz')
    header = nii_input.header
    arr = np.transpose(np.array(nii_input.dataobj), axes=[2, 1, 0])
    
    assert seg_old_size.shape == arr.shape
    
    raw_filename = npz.replace('.npz', '_gt1.raw')
    fileobj = open(raw_filename, mode='wb')
    off = np.array(seg_old_size, dtype=np.uint8)
    off.tofile(fileobj)
    fileobj.close()
    
    stats.append([raw_filename, thresholds])
    print(raw_filename)

stats_df = pd.DataFrame(stats, columns=['id', 'threshold'])
stats_df.to_csv('gmm_th.csv')
#pdb.set_trace()
    
