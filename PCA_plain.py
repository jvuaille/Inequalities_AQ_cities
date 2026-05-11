# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

def make_3D_plot_PCA(my_x, my_x_legend, my_y, my_y_legend, my_z, my_z_legend, my_scenario ,
                     my_color_legend, my_color_file_name, my_color_legend_name, my_angle, my_path):
    fig=plt.figure(figsize=(12,12))
    ax = plt.axes(projection='3d')
    my_plot=ax.scatter(my_x,my_y,my_z, c=my_color_legend, marker='o', cmap='magma_r') 
    col=plt.colorbar(my_plot, shrink=0.8, label=my_color_legend_name)
    col.ax.set_ylabel('\n%s'%my_color_legend_name, fontsize=12)
    col.ax.tick_params(labelsize=12)
    ax.set_zlabel('\n%s'%my_z_legend, fontsize=12)
    ax.view_init(my_angle[0], my_angle[1])
    plt.ylabel('\n%s'%my_y_legend, fontsize=12)
    plt.xlabel('\n%s'%my_x_legend, fontsize=12)
    plt.tick_params(labelsize=12, axis='both', which='major')
#    plt.axis('equal')
    plt.tight_layout()
    fig.savefig((my_path+'\\'+'%s_%s.png'%(my_color_file_name, my_scenario))) 


def make_2D_plot_PCA(my_x, my_x_legend, my_y, my_y_legend, my_scenario ,my_color_legend, my_color_file_name, my_color_legend_name, my_path):
    fig=plt.figure(figsize=(12,12), dpi=500)
    my_plot=plt.scatter(my_x,my_y, c=my_color_legend, marker='o', cmap='magma_r') 
    col=plt.colorbar(my_plot, shrink=0.8, label=my_color_legend_name)
    col.ax.set_ylabel('\n%s'%my_color_legend_name, fontsize=12)
    col.ax.tick_params(labelsize=12)
    plt.ylabel('\n%s'%my_y_legend, fontsize=12)
    plt.xlabel('\n%s'%my_x_legend, fontsize=12)
    plt.tick_params(labelsize=12, axis='both')
    plt.axis('equal')
    plt.tight_layout()
    fig.savefig((my_path+'\\'+'%s_%s.png'%(my_color_file_name, my_scenario))) 

def make_circle_corr(my_data, my_features_names, my_pc_x, my_pc_x_legend, my_pc_y, my_pc_y_legend, my_eigen_values, my_eigen_vectors, my_scenario, my_file_name, my_path):
    fig=plt.figure(figsize=(10,10))
    plt.Circle((0,0),radius=1, color='g', fill=False)
    circle1=plt.Circle((0,0),radius=1, color='g', fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    for idx in range(len(my_eigen_values)):
    	x = np.sqrt(my_eigen_values[my_pc_x])*my_eigen_vectors[idx,my_pc_x]
    	y = np.sqrt(my_eigen_values[my_pc_y])*my_eigen_vectors[idx,my_pc_y]
    	plt.plot([0.0,x],[0.0,y])
    	plt.plot(x, y, 'rx')
    	plt.annotate(my_features_names[idx], xy=(x,y))
    plt.tick_params(labelsize=9, axis='both', which='major')
    plt.xlabel(my_pc_x_legend)
    plt.ylabel(my_pc_y_legend)
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.title("Circle of Correlations")
    fig.savefig((my_path+'\\'+'circle_corr_%s_%s.png'%(my_file_name,my_scenario))) 

def stat_data_Ml(my_x, my_x_legend, my_y, my_y_legend, my_scenario, file_name, my_path):
    fig=plt.figure(figsize=(8,8))
    plt.plot(my_x, my_y.values[1,:], label='Average')
    plt.plot(my_x, my_y.values[2,:], label='Standard deviation')
    plt.plot(my_x, my_y.values[4,:], label='First quartile')
    plt.plot(my_x, my_y.values[5,:], label='Second quartile')
    plt.plot(my_x, my_y.values[6,:], label='Third quartile')
    plt.xlabel(my_x_legend, fontsize=20)
    plt.ylabel(my_y_legend, fontsize=20)
    plt.xticks(np.arange(1, len(my_x)+1,1))
    plt.ylim(-5,5)
    plt.tick_params(labelsize=20, axis='both', which='major')
    plt.legend(fontsize=20, loc='upper right')
    fig.savefig((my_path+'\\'+'%s_%s.png'%(file_name,my_scenario))) 


# =============================================================================
# Perform a PCA
# =============================================================================
# each variable/feature has a new set of coordinates in the new dimensions,
# where the first dimension is given by the first PC etc.
# the components give the matrix (n components, n features) where we have the coordinates of each feature 
# as column in the new dimensions

def PC_analysis(my_features, my_nb_components, 
                my_scenario, variables_names_list,
                my_outputs_folder_path):
    pca = PCA(n_components=my_nb_components)
    pca.fit_transform(my_features)

    ### loadings ca.components_ with size (n_components, n_features) transposed (n_features, n_components)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_) # each feature vector with coordinates on each C is multiplied by eigen values. get the 'weight or load' of each feature on each C
    pd.DataFrame(data=loadings, columns=np.arange(0, len(loadings)), index=variables_names_list).to_csv(
                my_outputs_folder_path+'\\'+'Contrib_method2_%s.csv' %(
                my_scenario), index=True)    
    
    ### loadings to plot 
    fig=plt.figure(figsize=(25,8))
    plt.bar(np.arange(1, len(pca.components_)*2,2), np.abs(loadings.T[0]), width=1.5, 
            align='center', label='PC %d'%(1))
    plt.bar(np.arange(1, len(pca.components_)*2,2), np.abs(loadings.T[1]), width=1.5, 
            align='center', label='PC %d'%(2), bottom = np.array(np.abs(loadings.T[0])))
    plt.bar(np.arange(1, len(pca.components_)*2,2), np.abs(loadings.T[2]), width=1.5, 
            align='center', label='PC %d'%(3), bottom = np.array(np.abs(loadings.T[0]))
            +np.array(np.abs(loadings.T[1])))
    plt.bar(np.arange(1, len(pca.components_)*2,2), np.abs(loadings.T[3]), width=1.5, 
            align='center', label='PC %d'%(4), bottom = np.array(np.abs(loadings.T[0]))
            +np.array(np.abs(loadings.T[1]))+np.array(np.abs(loadings.T[2])))
    plt.xticks(np.arange(1, len(pca.components_)*2,2),variables_names_list, 
               fontsize=12, rotation=90)
    plt.ylabel('Contribution of features on PCs\n', fontsize=15)
    plt.xlabel('Feature', fontsize=15)
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=15, axis='both', which='major')
    # plt.ylim(bottom=0)
    plt.xlim(left=0.5)
    plt.tight_layout()
    fig.savefig(my_outputs_folder_path+'\\'+'PCs_loadings_%s.png' %(my_scenario))
    
    ### eigen values on each PC
    index_pc=np.arange(1,len(pca.explained_variance_)+1)
    fig=plt.figure(figsize=(15,8))
    plt.bar(index_pc, pca.explained_variance_, width=0.5, align='center', color='limegreen')
    plt.title(np.array_str(pca.explained_variance_))
    plt.xticks(np.arange(1, len(pca.explained_variance_)+1,1))
    plt.ylabel('Variance (eigen value)\n', fontsize=15)
    plt.xlabel('\nPrincipal component index', fontsize=15)
    plt.tick_params(labelsize=15, axis='both', which='major')
    plt.ylim(bottom=0, top=pca.explained_variance_[0]+0.5)
    plt.tight_layout()
    fig.savefig(my_outputs_folder_path+'\\'+'eigen_values_%s.png' %(my_scenario))
    
    cum_var=np.cumsum(pca.explained_variance_/np.sum(pca.explained_variance_)*100)
    where_80perc=np.where(cum_var >= 80)[0][0]
    where_90perc=np.where(cum_var >= 90)[0][0]
    where_95perc=np.where(cum_var >= 95)[0][0]
    where_99perc=np.where(cum_var >= 99)[0][0]
    print('To capture 90% of the variance in the dataset we need at least '+
          '%d PCs, %d for at least'%(index_pc[where_90perc],index_pc[where_95perc])+
          ' 95% of the variance, ' +'%d'%(index_pc[where_80perc])+' for at least 80% of the variance and '+'%d'%index_pc[where_99perc] +
          ' for at least 99% of the variance.')

    ### project features on x PCs
    my_projected_features=np.dot(pca.components_,my_features.T)[0:my_nb_components]
    pd.DataFrame(data=pca.components_).to_csv(my_outputs_folder_path+'\\'+
                'PCs_%s.csv' %(my_scenario), index=False,index_label=False)
        
    # =============================================================================
    # circle of correlations
    # =============================================================================
    make_circle_corr(my_features, variables_names_list, 0, 'PC 1', 1, 'PC 2', pca.explained_variance_,
                     pca.components_.T, my_scenario, 'PC_1_2', my_outputs_folder_path)
    make_circle_corr(my_features, variables_names_list, 2, 'PC 3', 3, 'PC 4', pca.explained_variance_,
                     pca.components_.T, my_scenario, 'PC_3_4', my_outputs_folder_path)
    
    # =============================================================================
    # Statistics on training, validating and testing subsets
    # =============================================================================
    dataset=pd.DataFrame(data=my_projected_features.T).describe()
    
    stat_data_Ml(np.arange(1,len(my_projected_features)+1), "PC's number", dataset, 
                 "Principal component statistics - Full dataset", 
                 my_scenario, 'Full dataset', my_outputs_folder_path)
    
    return my_projected_features, pca.components_, index_pc[where_80perc] # return the number of components for 80% of the variance explained
