# Kay Hutchinson 8/9/22
# Compare gesture transcripts in terms of acc, edit score, f1, iou (micro and macro)

import pandas as pd
import numpy as np
import os
import os
import sys
import glob
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Group all rows with the same MP and return as a new df as <start, end, MP>
def group(dfMP):
    # Find start and end indices of each group of rows with the same context label
    dfMP['subgroup'] = (dfMP['Y'] != dfMP['Y'].shift(1)).cumsum()
    #print(dfMP)

    # Create 'subgroup' column indicating groups of consecutive rows with same MP label
    dfGrouped = dfMP.groupby('subgroup').apply(lambda x: (x['Sample'].iloc[0], x['Sample'].iloc[-1], x['Y'].iloc[0]))
    #print(dfGrouped)

    # Split list from lambda function into columns again
    dfGrouped = pd.DataFrame(dfGrouped.tolist(), index=dfGrouped.index)
    #print(dfGrouped)

    return dfGrouped

# Convert list of labels to a transcript (intermediate step uses dataframes)
def listToTranscript(list):
    dfMP = pd.DataFrame(list, columns=["Y"])
    dfMP.insert(0, 'Sample', range(0, len(list)))

    # Group MPs in a dataframe with start, end, and MP label
    mps = group(dfMP)

    # convert MPs dataframe to list
    transcript = mps.values.tolist()

    return transcript

# Convert transcript to list of labels
def transcriptToList(transcript):
    list = []
    #print(transcript)
    # For each label, fill in the list with that label from start to end sample number
    for t in transcript:
        fill = [t[2]]*(int(t[1])-int(t[0])+1)
        list[int(t[0]):int(t[1])+1] = fill

    return list

# Convert transcript to sequence (one way conversion)
def transcriptToSequence(transcript):
    sequence = []
    for i in transcript:
        sequence.append(i[2])
    return sequence

# Read an MP transcript from a given file path
def readMPTranscript(filePath):
    # Read in file
    with open(filePath) as f:
        lines = f.readlines()
    # Drop header
    lines = lines[1:]

    # Reformat and take only the verb
    for i in range(len(lines)):
        lines[i] = lines[i].strip().split("(")[0]
        lines[i] = lines[i].split(" ")

    # Return MP transcript
    return lines


# Read a gesture transcript from a given file path
def readGTranscript(filePath):
    # Read in file
    with open(filePath) as f:
        lines = f.readlines()

    # Reformat and take only the verb
    for i in range(len(lines)):
        lines[i] = lines[i].strip().split("(")[0]
        lines[i] = lines[i].split(" ")

    # Return gesture transcript
    return lines


# Calculate edit score
def levenstein_(p,y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float64)
    for i in range(m_row+1):
        D[i,0] = i
    for i in range(n_col+1):
        D[0,i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1]==p[i-1]:
                D[i,j] = D[i-1,j-1]
            else:
                D[i,j] = min(D[i-1,j]+1,
                             D[i,j-1]+1,
                             D[i-1,j-1]+1)

    if norm:
        score = (1 - D[-1,-1]/max(m_row, n_col) ) * 100
    else:
        score = D[-1,-1]

    return score


# Visualize MP segments
def drawLists(list1, list2):
    # Cast list to df
    df1 = pd.DataFrame(list1, columns = ['labels'])
    df2 = pd.DataFrame(list2, columns = ['labels'])

    # Encode labels and add to df
    le = preprocessing.LabelEncoder()
    le.fit(df1['labels'])
    df1['encoded'] = le.transform(df1['labels'])

    le.fit(df2['labels'])
    df2['encoded'] = le.transform(df2['labels'])

    # Get list of classes from encoding
    le_mapping = list(le.classes_)

    # Look at sequences
    #print(transcriptToSequence(listToTranscript(list1)))
    #print(transcriptToSequence(listToTranscript(list2)))


    # Get color range for vmin and vmax in pcolorfast
    cmin = min(min(df1['encoded'].unique()), min(df2['encoded'].unique()))
    cmax = max(max(df1['encoded'].unique()), max(df2['encoded'].unique()))

    # Graph
    fig, axs = plt.subplots(2, 1)  #create two subplots

    # First plot
    ax = axs[0]
    c = ax.pcolorfast(df1.index[:], ax.get_ylim(), df1['encoded'].values[np.newaxis], cmap='Set1', alpha=0.3, vmin=cmin, vmax=cmax)
    ax.set_title('Left')

    # Labels and legend
    ax.set_xlabel('Frame')
    ax.set_ylabel('Motion Primitive')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.set_yticklabels(le_mapping)

    # Second plot
    ax = axs[1]
    c = ax.pcolorfast(df2.index[:], ax.get_ylim(), df2['encoded'].values[np.newaxis], cmap='Set1', alpha=0.3, vmin=cmin, vmax=cmax)
    ax.set_title('Left (K)')

    # Labels and legend
    ax.set_xlabel('Frame')
    ax.set_ylabel('Motion Primitive')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.set_yticklabels(le_mapping)

    font = {'size'   : 20}

    # Show plot
    fig.tight_layout()
    plt.show()


    '''
    # Old code from Zoey, also plots lines/levels for different labels
    plt.subplot(2, 1, 1)


    ax=df1['encoded'].plot(label='MP') #.apply(lambda x:x+6).plot(label = 'MP')

    print(df1['encoded'].values[np.newaxis])
    print(df2['encoded'].values[np.newaxis])

    #ax.set_xticks(np.arange(855, 855+len(table_process['Motion Primative_c'][855:2410]),150))
    #ax.set_xticklabels(np.arange(855//30,(855+len(table_process['Motion Primative_c'][855:2410]))//30,5))
    ax.pcolorfast(df1.index[:], ax.get_ylim(),df1['encoded'].values[np.newaxis],cmap='Set1', alpha=0.3)
    #ax.set_xlabel('Time (sec)',fontname='Arial')
    #ax.set_ylabel('Motion Primitive/Gesture',fontname='Arial',fontsize=20)
    #ax.set_yticks(np.arange(0,14))
    #ax.set_yticklabels(["Grasp","Pull","Push","Release","Touch","Untouch","G1","G11","G2",\
                        #"G3","G4","G5","G6","G8"])
    ax.legend()
    #ax.set_ylim([-0.2,13.2])
    #font = {'size'   : 20}

    #matplotlib.rc('font', **font)
    #fig.show()

    plt.subplot(2, 1, 2)
    ax=df2['encoded'].plot(label='MP') #.apply(lambda x:x+6).plot(label = 'MP')
    ax.pcolorfast(df2.index[:], ax.get_ylim(),df2['encoded'].values[np.newaxis],cmap='Set1', alpha=0.3)
    ax.legend()

    plt.show()
    '''
    #print(df)


# Visualize gesture segments
def drawListsGestures(list1, list2):
    # Cast list to df
    df1 = pd.DataFrame(list1, columns = ['labels'])
    df2 = pd.DataFrame(list2, columns = ['labels'])

    # Encode labels and add to df
    le = preprocessing.LabelEncoder()
    le.fit(df1['labels'])
    df1['encoded'] = le.transform(df1['labels'])

    le.fit(df2['labels'])
    df2['encoded'] = le.transform(df2['labels'])

    # Get list of classes from encoding
    le_mapping = list(le.classes_)

    # Look at sequences
    #print(transcriptToSequence(listToTranscript(list1)))
    #print(transcriptToSequence(listToTranscript(list2)))

    # Get color range for vmin and vmax in pcolorfast
    cmin = min(min(df1['encoded'].unique()), min(df2['encoded'].unique()))
    cmax = max(max(df1['encoded'].unique()), max(df2['encoded'].unique()))+1
    
    # Graph
    fig, axs = plt.subplots(2, 1)  #create two subplots

    # First plot
    ax = axs[0]
    c = ax.pcolorfast(df1.index[:], ax.get_ylim(), df1['encoded'].values[np.newaxis], cmap='Set1', alpha=0.3, vmin=cmin, vmax=cmax)
    ax.set_title('Gestures (gt)')

    # Labels and legend
    ax.set_xlabel('Frame')
    ax.set_ylabel('Gestures (gt)')
    cbar = fig.colorbar(c, ax=ax, ticks = range(cmax-cmin-1))
    #cbar.ax.set_yticklabels(le_mapping)

    tick_locs = (np.arange(cmax) + 0.5)*(cmax-1)/cmax
    cbar.set_ticks(tick_locs)
    # set tick labels (as before)
    cbar.set_ticklabels(le_mapping)


    # Second plot
    ax = axs[1]
    c = ax.pcolorfast(df2.index[:], ax.get_ylim(), df2['encoded'].values[np.newaxis], cmap='Set1', alpha=0.3, vmin=cmin, vmax=cmax)
    ax.set_title('Gestures (cc)')

    # Labels and legend
    ax.set_xlabel('Frame')
    ax.set_ylabel('Gestures (cc)')
    cbar = fig.colorbar(c, ax=ax, ticks = range(cmax-cmin-1))
    #cbar.ax.set_yticklabels(le_mapping)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(le_mapping)

    font = {'size'   : 20}

    # Show plot
    fig.tight_layout()
    plt.show()


    '''
    # Old code from Zoey, also plots lines/levels for different labels
    plt.subplot(2, 1, 1)


    ax=df1['encoded'].plot(label='MP') #.apply(lambda x:x+6).plot(label = 'MP')

    print(df1['encoded'].values[np.newaxis])
    print(df2['encoded'].values[np.newaxis])

    #ax.set_xticks(np.arange(855, 855+len(table_process['Motion Primative_c'][855:2410]),150))
    #ax.set_xticklabels(np.arange(855//30,(855+len(table_process['Motion Primative_c'][855:2410]))//30,5))
    ax.pcolorfast(df1.index[:], ax.get_ylim(),df1['encoded'].values[np.newaxis],cmap='Set1', alpha=0.3)
    #ax.set_xlabel('Time (sec)',fontname='Arial')
    #ax.set_ylabel('Motion Primitive/Gesture',fontname='Arial',fontsize=20)
    #ax.set_yticks(np.arange(0,14))
    #ax.set_yticklabels(["Grasp","Pull","Push","Release","Touch","Untouch","G1","G11","G2",\
                        #"G3","G4","G5","G6","G8"])
    ax.legend()
    #ax.set_ylim([-0.2,13.2])
    #font = {'size'   : 20}

    #matplotlib.rc('font', **font)
    #fig.show()

    plt.subplot(2, 1, 2)
    ax=df2['encoded'].plot(label='MP') #.apply(lambda x:x+6).plot(label = 'MP')
    ax.pcolorfast(df2.index[:], ax.get_ylim(),df2['encoded'].values[np.newaxis],cmap='Set1', alpha=0.3)
    ax.legend()

    plt.show()
    '''
    #print(df)



# Visualize mp and gesture segments
def drawListsMPsandGestures(list1, list2):
    # Cast list to df
    df1 = pd.DataFrame(list1, columns = ['labels'])
    df2 = pd.DataFrame(list2, columns = ['labels'])

    # Encode labels and add to df
    le = preprocessing.LabelEncoder()
    le.fit(df1['labels'].append(df2['labels']))
    df1['encoded'] = le.transform(df1['labels'])

    #le.fit(df2['labels'])
    df2['encoded'] = le.transform(df2['labels'])

    # Get list of classes from encoding
    le_mapping = list(le.classes_)

    # Look at sequences
    #print(transcriptToSequence(listToTranscript(list1)))
    #print(transcriptToSequence(listToTranscript(list2)))

    # Get color range for vmin and vmax in pcolorfast
    cmin = min(min(df1['encoded'].unique()), min(df2['encoded'].unique()))
    cmax = max(max(df1['encoded'].unique()), max(df2['encoded'].unique()))+1
    
    # Graph
    fig, axs = plt.subplots(2, 1)  #create two subplots

    # First plot
    ax = axs[0]
    c = ax.pcolorfast(df1.index[:], ax.get_ylim(), df1['encoded'].values[np.newaxis], cmap='Set1', alpha=0.3, vmin=cmin, vmax=cmax)
    ax.set_title('Gestures (gt)')

    # Labels and legend
    #ax.set_xlabel('Frame')
    #ax.set_ylabel('Gestures (gt)')
    cbar = fig.colorbar(c, ax=ax, ticks = range(cmax-cmin-1))
    #cbar.ax.set_yticklabels(le_mapping)

    tick_locs = (np.arange(cmax) + 0.5)*(cmax-1)/cmax
    cbar.set_ticks(tick_locs)
    # set tick labels (as before)
    cbar.set_ticklabels(le_mapping)


    # Second plot
    ax = axs[1]
    c = ax.pcolorfast(df2.index[:], ax.get_ylim(), df2['encoded'].values[np.newaxis], cmap='Set1', alpha=0.3, vmin=cmin, vmax=cmax)
    ax.set_title('MPs')

    # Labels and legend
    ax.set_xlabel('Frame')
    #ax.set_ylabel('MPs')
    cbar = fig.colorbar(c, ax=ax, ticks = range(cmax-cmin-1))
    #cbar.ax.set_yticklabels(le_mapping)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(le_mapping)

    font = {'size'   : 20}

    # Show plot
    fig.tight_layout()
    plt.show()






# Helper for getTransitionMatrix()
# create a function that maps transition probability dataframe
# to markov edges and weights
def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx,col)] = Q.loc[idx,col]
    return edges




# Given all the gesture transcripts in a task, sum up the transitions in a matrix
#   To:  G1      G2    G3      G4   G5   ...   From:
# Q = [   .       .     .       .    .    .,    G1
#         .       .     .       .    .    .,    G2
#         .       .     .       .    .    .,    G3
#         .       .     .       .    .    .,    G4
#         .       .     .       .    .    .,    G5
#         .       .     .       .    .    ., ]  ...
def getTransitionMatrixGestures(mpDir):
    # based on http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017
    import networkx as nx
    import matplotlib.pyplot as plt
    from pprint import pprint
    import graphviz as gv

    # MPs and encodings
    Gs = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12", "G13", "G14", "G15"]
    encGs = {"G1":0, "G2":1, "G3":2, "G4":3, "G5":4, "G6":5, "G7":6, "G8":7, "G9":8, "G10":9, "G11":10, "G12":11, "G13":12, "G14":13, "G15":14}

    # Define threshold for edge probabilities to determine which edges are shown in the graph
    edgeThreshold = 0.1

    # Initialize transition matrix
    Q = np.zeros((len(Gs), len(Gs)))

    # Loop through transcripts and create Q
    files = glob.glob(mpDir+"/*.txt")
    for file in files:
        # get file name of transcript
        trial = file.split("/")[-1]
        #print(trial)

        # Read in transcript
        with open(file, 'r') as t:
            lines = t.readlines()
            #print(lines)
            for i in range(len(lines)-1):
                #print(i)
                line = lines[i].split()
                # get gesture from line
                g = line[2]
                # get next gesture from next line
                linen = lines[i+1].split()
                gn = linen[2]
                # Increment transition in Q matrix
                #print(Q[encMPs[mp], encMPs[mpn]])
                #print(encGs[g], encGs[gn])
                Q[encGs[g], encGs[gn]] = Q[encGs[g], encGs[gn]]+1
                #print(Q[encMPs[mp], encMPs[mpn]])
    #print(Q)

    # Normalize Q by row
    rowSums = Q.sum(axis=1)
    Q = np.nan_to_num(Q/rowSums[:,np.newaxis])
    #Q = np.nan_to_num(Q)
    #print(Q)


    # Convert np array to df
    dfQ = pd.DataFrame(Q, columns = Gs, index = Gs)
    print(dfQ)
    #print(dfQ.sum(axis=1))

    # Get edge weights
    edges_wts = _get_markov_edges(dfQ)
    #pprint(edges_wts)


    # Visualize and save figure with graphviz
    g = gv.Graph(format='png')
    dot = gv.Digraph('test2')
    for k, v in edges_wts.items():
        if v > edgeThreshold:     # draw edges with a probability greater than the defined threshold
            tmp_origin, tmp_destination = k[0], k[1]
            dot.edge(tmp_origin, tmp_destination, label='%.2f'%v)
    dot.format='png'#'svg'
    dot.render()




# Given all the context transcripts in a task, sum up the transitions in a matrix; dynamically adds next context
#   To:   1       2     3       4    5   ...   From:
# Q = [   .       .     .       .    .    .,    1 
#         .       .     .       .    .    .,    2
#         .       .     .       .    .    .,    3
#         .       .     .       .    .    .,    4
#         .       .     .       .    .    .,    5
#         .       .     .       .    .    ., ]  ...
def getTransitionMatrixContext(cDir):
    # based on http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017
    import networkx as nx
    import matplotlib.pyplot as plt
    from pprint import pprint
    import graphviz as gv

    #from context_to_gestures import condenseContext, group

    # MPs and encodings
    Cs = ["Start", "End"]
    encCs = {}

    # Define threshold for edge probabilities to determine which edges are shown in the graph
    edgeThreshold = 8 #0.1

    # Loop through transcripts and create Q
    files = glob.glob(cDir+"/*.txt")
    

    # Read through all files and get list of all context
    for file in files:
        print(file)
        
        # Read in transcript
        with open(file, 'r') as t:
            df = pd.read_csv(file, delimiter = " ", header=None)
            # condense all context var columns into a string in one column
            dfContext = condenseContext(df)
            
            # group rows with same context label
            dfGrouped = group(dfContext)
            
            clist = list(dfGrouped['Context'].unique())
            Cs = list(set(Cs+clist))
    
    print(Cs)
     # Initialize transition matrix
    Q = np.zeros((len(Cs), len(Cs))) 
    
    # Read through all files again and make transition matrix
    for file in files:
        # get file name of transcript
        #trial = file.split("/")[-1]
        #print(file)
        
        # Read in transcript
        with open(file, 'r') as t:
            df = pd.read_csv(file, delimiter = " ", header=None)
            # condense all context var columns into a string in one column
            dfContext = condenseContext(df)
            
            # group rows with same context label
            dfGrouped = group(dfContext)
            npGrouped = dfGrouped.to_numpy()
            
            # Increment start and end in Q
            fromI = Cs.index("Start")
            toI = Cs.index(npGrouped[0][2])
            Q[fromI, toI] = Q[fromI, toI]+1
            fromI = Cs.index(npGrouped[-1][2])
            toI = Cs.index("End")
            Q[fromI, toI] = Q[fromI, toI]+1

            for i in range(len(npGrouped)-1):
                #print(i)
                line = npGrouped[i]
                # get gesture from line
                c = line[2]
                # get next gesture from next line
                linen = npGrouped[i+1]
                cn = linen[2]
                
                fromI = Cs.index(c)
                toI = Cs.index(cn)
                # Increment transition in Q matrix
                Q[fromI, toI] = Q[fromI, toI]+1
    print(Q)
    
    # Normalize Q by row
    rowSums = Q.sum(axis=1)
    #Q = np.nan_to_num(Q/rowSums[:,np.newaxis])
    #Q = np.nan_to_num(Q)
    #print(Q)


    # Convert np array to df
    dfQ = pd.DataFrame(Q, columns = Cs, index = Cs)
    print(dfQ)
    #print(dfQ.sum(axis=1))

    # Get edge weights
    edges_wts = _get_markov_edges(dfQ)
    #pprint(edges_wts)


    # Visualize and save figure with graphviz
    g = gv.Graph(format='png')
    dot = gv.Digraph('context')
    for k, v in edges_wts.items():
        if v > edgeThreshold:     # draw edges with a probability greater than the defined threshold
            tmp_origin, tmp_destination = k[0], k[1]
            dot.edge(tmp_origin, tmp_destination, label='%.2f'%v)
    dot.format='png'#'svg'
    dot.render()



# Function to condense all context variables into one column for easier comparison
# Takes df of raw context transcript and concatenates state variables into one
# string for easier comparison, returned df structure is <frame, context>
def condenseContext(df):
    # number of columns in DataFrame
    numCols = len(df.columns)
    numContext = numCols - 1    # first column is frame number

    # new dataframe to hold frame and context
    dfContext = pd.DataFrame(columns = ['Frame'])
    dfContext['Frame'] = df.iloc[:,0]
    #print(dfContext)


    # concatenate states into one string
    # initialize with first context variable
    dfContext['Context'] = df.iloc[:,1]
    for i in range(2, numContext+1):
        #print(i)
        dfContext['Context'] = dfContext['Context'].map(str) + df.iloc[:,i].map(str)

    # If task is Knot Tying, convert 00003 into 00000 because the transition 00003 -> 00000 doesn't mean anything
    if task == "Knot_Tying":
        dfContext['Context'] = dfContext['Context'].replace(["00003"],"00000")

    # If task is Pea on a Peg, convert 00004 into 00000 because the transition 00004 -> 00000 doesn't mean anything
    # and 00004 is an invalid state because the pea state is only used if a pea is held
    if task == "Pea_on_a_Peg":
        dfContext['Context'] = dfContext['Context'].replace(["00004"],"00000")

    return dfContext


'''
# Function to group all rows with same context label and write into a new df as <start, end, context>
def group(dfContext):
    # find start and end indices of each group of rows with the same context label
    dfContext['subgroup'] = (dfContext['Context'] != dfContext['Context'].shift(1)).cumsum()
    
    # create 'subgroup' column indicating groups of consecutive rows with same 'Context' label
    #dfGrouped = dfContext.groupby('subgroup',as_index=False).apply(lambda x: (x['Frame'].iloc[0], x['Frame'].iloc[-1], x['Context'].iloc[0]))  #head(1)))
    dfGrouped = dfContext.groupby(['subgroup']).apply(lambda x: (x['Frame'].iloc[0], x['Frame'].iloc[-1], x['Context'].iloc[0]))
    #print(dfGrouped)

    # cast to df and return
    myDF = pd.DataFrame(dfGrouped.tolist(), columns=["Start", "End", "Context"])
    
    return myDF

'''



# Lea's utils code
def segment_labels(Yi):   # returns an array of labels, basically seq
    idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs)-1)])
    return Yi_split

def segment_intervals(Yi):   # returns start and end indices for each segment
    idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
    intervals = [(idxs[i],idxs[i+1]) for i in range(len(idxs)-1)]
    return intervals

def get_overlap_f1_colin(P, Y, n_classes=0, overlap=.1):
    
    def overlap_(p,y, n_classes, overlap):

        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]
        
        # We keep track of the per-class TPs, and FPs.
        # In the end we just sum over them though.
        TP = np.zeros(n_classes, np.float64)
        FP = np.zeros(n_classes, np.float64)
        true_used = np.zeros(n_true, np.float64)

        # Sum IoUs for each class
        IoUs = np.zeros(n_classes, np.float64)
        nIoUs = np.zeros(n_classes, np.float64)

        for j in range(min(n_pred,n_true)):
            #print(n_pred[j])
            # Compute IoU against all others
            intersection = np.minimum(pred_intervals[j,1], true_intervals[:,1]) - np.maximum(pred_intervals[j,0], true_intervals[:,0])
            union = np.maximum(pred_intervals[j,1], true_intervals[:,1]) - np.minimum(pred_intervals[j,0], true_intervals[:,0])
            IoU = (intersection / union)*(pred_labels[j]==true_labels)
            
            #print("G"+str(pred_labels[j]+1))
            #if not true_used[idx]:   # might need this?
            IoUs[pred_labels[j]] += max(max(IoU), 0)
            nIoUs[pred_labels[j]] += 1

            # Get the best scoring segment (index)
            idx = IoU.argmax()

            # If the IoU is high enough and the true segment isn't already used
            # Then it is a true positive. Otherwise is it a false positive.
            if IoU[idx] >= overlap and not true_used[idx]:
                TP[pred_labels[j]] += 1
                true_used[idx] = 1
            else:
                FP[pred_labels[j]] += 1
                #print("Wrong label?? at Pred: " + str(pred_intervals[j,0]) + " with IoU " + str(IoU[idx]) + " gcc thinks it's " + str(pred_labels[j]) + " but g says it's " + str(true_labels[j]))
                #print("GT: " + str(true_labels[j]))
            
            # Check for annotation errors
            if (true_labels[j] != pred_labels[j]) and (IoU[idx] < 0.1) and (notTransitionError(true_labels, pred_labels, j)):
                print("Frame: " + str(true_intervals[j,0]+listStart) + " gt: " + str(true_labels[j]+1) + " pred: " + str(pred_labels[j]+1) + " iou: " + str(IoU[idx]))

        TP = TP.sum()
        FP = FP.sum()
        # False negatives are any unused true segment (i.e. "miss")
        FN = n_true - true_used.sum()

        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        F1 = 2 * (precision*recall) / (precision+recall)  #RuntimeWarning: invalid value encountered in double_scalars

        # If the prec+recall=0, it is a NaN. Set these to 0.
        F1 = np.nan_to_num(F1)

        # Take average IoU of each class in this trial wrt number of correct labels
        classIoUs = (IoUs/nIoUs)
        classIoUs = np.nan_to_num(classIoUs)
        macroIoU = sum(classIoUs)/np.count_nonzero(nIoUs)

        # avg over trial:
        microIoU = np.sum(IoUs)/np.sum(nIoUs)
        #print(avgIoUs)
        #sys.exit()

        return F1*100, microIoU, macroIoU, nIoUs, classIoUs

    # if type(P) == list:
    #     return np.mean([overlap_(P[i],Y[i], n_classes, overlap) for i in range(len(P))])
    # else:
    return overlap_(P, Y, n_classes, overlap)



def notTransitionError(gt, pred, i):
  # Get transcripts
  transcriptgt = listToTranscript(gt)
  transcriptpred = listToTranscript(pred)
  # Find where i is in the transcripts and get current pred and gt gesture
  for t in range(len(transcriptgt)):
    line = transcriptgt[t]
    if (i >= line[0]) and (i <= line[1]):
      #print(line)
      #print("prev g: " + str(transcriptgt[max(t-1,0)]) + " curr g: " + str(transcriptgt[t]) + " next g: " + str(transcriptgt[min(t+1, len(transcriptgt)-1)]))
      prevgt = transcriptgt[max(t-1,0)][2]
      currgt = transcriptgt[t][2]
      nextgt = transcriptgt[min(t+1,len(transcriptgt)-1)][2]
  for t in range(len(transcriptpred)):
    line = transcriptpred[t]
    if (i >= line[0]) and (i <= line[1]):
      #print(line)
      #print("prev g: " + str(transcriptpred[max(t-1,0)]) + " curr g: " + str(transcriptpred[t]) + " next g: " + str(transcriptpred[min(t+1, len(transcriptpred)-1)]))
      prevpred = transcriptpred[max(t-1,0)][2]
      currpred = transcriptpred[t][2]
      nextpred = transcriptpred[min(t+1,len(transcriptpred)-1)][2]
  if (currpred == prevgt) or (currpred == nextgt):
    #print("Just a boundary error...")
    return False

  return True




if __name__ == "__main__":

    # Get task from command line
    task = "Knot_Tying"

    source1 = "automated"
    source2 = "consensus"

    #baseDir = os.path.dirname(os.getcwd())
    baseDir = os.getcwd()
    # Transcript and video directories
    taskDir = os.path.join(baseDir, "Datasets", "dV", task)
    gDir = os.path.join(taskDir, "gestures") #"gestures")
    gccDir = os.path.join(taskDir, "gestures_consensus_context")
    gacDir = os.path.join(taskDir, "gestures_automated_context")
    mpDir = os.path.join(taskDir, "motion_primitives_baseline")
    cDir = os.path.join(taskDir, "transcriptions")

    # set up comparisons based on input
    if source1 == "automated":
        dir1 = gacDir
    elif source1 == "consensus":
        dir1 = gccDir
    elif source1 == "original":
        dir1 = gDir
    elif source1 == "K" and task == "Needle_Passing":
        dir1 = os.path.join(taskDir, "gestures_K") 
    else:
        print("invalid context source")
        sys.exit()
    
    # set up comparisons based on input
    if source2 == "automated":
        dir2 = gacDir
    elif source2 == "consensus":
        dir2 = gccDir
    elif source2 == "original":
        dir2 = gDir
    elif source2 == "K" and task == "Needle_Passing":
        dir2 = os.path.join(taskDir, "gestures_K") 
    else:
        print("invalid context source")
        sys.exit()



    '''
    # Read context transcripts and make the state machine
    #getTransitionMatrixGestures(gccDir)
    getTransitionMatrixContext(cDir)
    sys.exit()
    '''

    # Counters
    tacc = 0
    tedit = 0
    tn = 0
    tf1 = 0
    tmicroiou = 0
    tmacroiou = 0
    tnious = 0
    tclassious = 0

    bestacc = 0
    worstacc = 100
    bestedit = 0
    worstedit = 100


    # list of files, loop through list with least files (since test set for automated labels is small)
    dir1Files = os.listdir(dir1)
    dir2Files = os.listdir(dir2)
    if len(dir1Files) < len(dir2Files):
        allFiles = dir1Files 
    else:
        allFiles = dir2Files

    # For each transcript
    for f in allFiles: #os.listdir(dir1): #[0:1]:
        print(f)

        # -------------------------------------
        '''
        # Draw mp and gesture transcript
        gPath = os.path.join(gDir, f)
        mpPath = os.path.join(mpDir, f)
        linesg = readGTranscript(gPath)
        linesmp = readMPTranscript(mpPath)
        listg = transcriptToList(linesg)
        listmp = transcriptToList(linesmp)
        drawListsMPsandGestures(listg, listmp)
        sys.exit()
        '''
        # --------------------------------------

        # Paths to the transcripts to compare
        path1 = os.path.join(dir1, f)
        path2 = os.path.join(dir2, f)

        # Read in transcripts
        lines1 = readGTranscript(path1)
        lines2 = readGTranscript(path2)

        # Convert to lists
        list1 = transcriptToList(lines1)
        list2 = transcriptToList(lines2)
        # print(lines1)
        # print(lines2)

        # Trim g lists because JIGSAWS starts late and ends early and this effects the metrics
        list1Start = int(lines1[0][0])
        list1End = int(lines1[-1][1])
        list2Start = int(lines2[0][0])
        list2End = int(lines2[-1][1])
        listStart = max(list1Start, list2Start)
        listEnd = min(list1End, list2End)
        
        list1 = list1[listStart-list1Start:-(list1End-listEnd+1)]
        list2 = list2[listStart-list2Start:-(list2End-listEnd+1)]
        
        print(lines1)
        print(lines2)

        #drawLists(list1, list2)
        #sys.exit()


        # Get accuracy
        acc = (np.mean([np.mean(list1[i]==list2[i]) for i in range(min(len(list1), len(list2)))])*100)
        #print(acc)

        # Get edit score
        lines1 = listToTranscript(list1)
        lines2 = listToTranscript(list2)  # combine consecutive identical gesture labels (increases F1 score)
        seq1 = transcriptToSequence(lines1)
        seq2 = transcriptToSequence(lines2)
        edit = (levenstein_(seq1, seq2, norm=True))
        #print(edit)

        tn = tn+1
        tacc = tacc+acc
        tedit = tedit+edit
        print(acc)
        print(edit)
        if acc > bestacc:
            bestacc = acc
        if edit > bestedit:
            bestedit = edit
        if acc < worstacc:
            worstacc = acc
        if edit < worstedit:
            worstedit = edit

        # Encode gesture labels to numbers by dropping the 'G' to use Lea's code
        # also subtract 1 from each label so it can be used for indexing
        listG1 = [int(list1[g][1:])-1 for g in range(len(list1))]
        listG2 =  [int(list2[g][1:])-1 for g in range(len(list2))]

        # Get number of unique classes in the labels 
        n_classes = 15 #max(max(np.unique(listG)), max(np.unique(listGCC)))+1
        
        # Get IoUs
        #ious = IOUs(listg, listgcc)
        # args: (P, Y, n_classes=0, overlap=.1)
        f1, microIoU, macroIoU, nious, classious = get_overlap_f1_colin(listG1, listG2, n_classes, 0.1)
        tf1 = tf1 + f1
        #print(ious)
        tmicroiou = tmicroiou + microIoU
        tmacroiou = tmacroiou + macroIoU
        tnious = tnious + nious
        tclassious = tclassious + classious

        #drawLists(listg, listgcc)

    print("Avg acc:  " + str(tacc/tn) + "\tBest acc:  " + str(bestacc) + "\tworst acc:  " + str(worstacc))
    print("Avg edit: " + str(tedit/tn) + "\tBest edit: " + str(bestedit) + "\tworst edit: " + str(worstedit))
    print("Avg f1: " + str(tf1/tn))
    print("Avg micro iou: " + str(tmicroiou/tn))
    print("Avg macro iou: " + str(tmacroiou/tn))
    my_formatted_list = [ '%.2f' % elem for elem in np.nan_to_num(tclassious/tn)]
    print("Avg class ious: " + str(my_formatted_list))
    #print("Best acc: " + str(bestacc) + " worst acc: " + str(worstacc))
    #print("Best edit " + str(bestedit) + " worst edit: " + str(worstedit))