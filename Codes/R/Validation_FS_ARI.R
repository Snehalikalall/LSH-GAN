
#Choice the Feature Selection method (FS) PCA_Loading, Fano Index, CV2 Index, Top 100 feature will be selected.
FS_clust<-function(PData,orgdata,nclass,genes_data){
	#####for PCA loading
#	   I=PCA_loading(PData) 
	    
	#####for Fano index
#	I=Fano_ind(PData)
    
### for CV2 index
          I= CV2(PData)
    
    
    datafilt = orgdata[, I]
    ###clustering using K-means
      #  cl <- kmeans(datafilt, nclass, nstart = 25)
    ###clustering using SC3
        cl <- SC3_clust(orgdata,I,orgclass,genes_data)
    return(cl)
        }



##############  FS modules ##################### 

PCA_loading<-function(PData){
	    PCA = prcomp(PData, scale = TRUE, center = TRUE) # Change variable name 'Data'
    Loadings = abs(PCA$rotation)
        Genes = vector()
        for (i in 1:dim(Loadings)[1])
		    {
			            Genes[i] = max(Loadings[i, 1:3])
	    }

	    I = order(Genes, decreasing = TRUE)[1:100]
	    return(I)
	        }

Fano_ind<-function(PData){
	    library(resample)
    bdata=as.matrix(PData)
        Fano_factor = colVars(bdata)/colMeans(bdata)
        ID = order(Fano_factor)
	    SelectedGenes_ID = ID[1:100]
	    return(SelectedGenes_ID)
	        }


CV2<-function(PData){
	#library(dplyr)
	library(plyr)
	library(data.table)
	#library('edgeR')
	    
	get_variable_gene<-function(m) {
		  
		  df<-data.frame(mean=colMeans(m),cv=apply(m,2,sd)/colMeans(m),var=apply(m,2,var))
	  df$dispersion<-with(df,var/mean)
	    df$mean_bin<-with(df,cut(mean,breaks=c(-Inf,quantile(mean,seq(0.1,1,0.05)),Inf)))
	    var_by_bin<-ddply(df,"mean_bin",function(x) {
				          data.frame(bin_median=median(x$dispersion),
						                    bin_mad=mad(x$dispersion))
					    })
	      df$bin_disp_median<-var_by_bin$bin_median[match(df$mean_bin,var_by_bin$mean_bin)]
	      df$bin_disp_mad<-var_by_bin$bin_mad[match(df$mean_bin,var_by_bin$mean_bin)]
	        df$dispersion_norm<-with(df,abs(dispersion-bin_disp_median)/bin_disp_mad)
	        df
	}


	datan=PData
	ngenes_keep = 100 #top 1000 genes
	cat("Select variable Genes...\n")
	df<-get_variable_gene(datan)
	gc()
	cat("Sort Top GenViewes...\n")
	disp_cut_off<-sort(df$dispersion_norm,decreasing=T)[ngenes_keep]
	cat("Cutoff Genes...\n")
	df$used<-df$dispersion_norm >= disp_cut_off
	top_features = head(order(-df$dispersion_norm),ngenes_keep)
	    return(top_features)
	    }
    

#######  SC3 clustering module  #######.

SC3_clust<-function(orgdata,I,orgclass,genes_data){
	library(SingleCellExperiment)
	#library(scater)
	#library(tibble)
	library(SC3)
	library(mclust)

	   ProcessedData=as.matrix(t(orgdata))
	   k=length(unique(as.matrix(orgclass)))


	       fea=as.matrix(I)
	       feaselData = as.matrix(ProcessedData[fea,])
	           biasegene=as.matrix(genes_data)
	           rownames(feaselData)=as.matrix(biasegene[fea])
		       colnames(feaselData)=as.matrix(orgclass)
		       write.csv(feaselData, "feanew.csv")
		           feanewdata= read.table("feanew.csv", header = TRUE,sep=',',row.names = 1)
		           rownames(orgclass)=colnames(feanewdata)
			       colnames(orgclass)=c('Cell_type1')
			       write.csv(orgclass, "organnot.csv")
			           annot= read.table("organnot.csv", header = TRUE,sep=',',row.names = 1)
			           
			           
			           sce <- SingleCellExperiment(
							             assays = list(
										           counts = as.matrix(feanewdata),
											           logcounts = as.matrix(feanewdata)
											         ), 
							             colData = annot
								         )
				       
				       # define feature names in feature_symbol column
				       rowData(sce)$feature_symbol <- rownames(sce)
				       
				       sclus<- sc3(sce, ks = k, gene_filter = FALSE,rand_seed = 1,biology = TRUE)
				           
				           
				           Annotations = sclus@colData@listData[[2]]
				           Annotations = as.matrix(Annotations)  

					   return(Annotations)
}


library(mclust)
# Path Setting
#setwd('/export/scratch2/sumanta/~IJCAI')
#Preprocessed Dataset Name Default yan
str="kleind" 
ari=vector()
orgdata=as.matrix(read.table(paste(str,'.csv',sep=""), header =FALSE,sep=','))
# Dataset Cell Type
orgclass=read.csv(paste(str,'_','annotation','.csv',sep=''))
nclass=length(unique(orgclass$X))
# Dataset Gene names
genes_data=as.matrix(read.csv(paste(str,'_','genes','.csv',sep=''),header=FALSE,sep=','))
for (p in 1:6){
# Read the generated data from gan (# "yan_mixdata_1.csv"), iter (1) is for Number of generated sample size (0.25p,0.5p,0.75p,1p,1.25p,1.5p) p is feature size
  str1=  "data_for_ari/kleind"   
  fname=paste(str1,'_','mixdata_iter',(p-1),'.csv',sep="")
    syndata=as.matrix(read.table(fname, header =FALSE,sep=','))
       syndata1= rbind(orgdata,syndata)
       print(dim(syndata1))
	    PData =syndata1# syndata1 #change this as syndata1 for iteration data and orgdata for original data
	    print(dim(PData))
	        
	        ####FS and clustering module##############
	        cl<-FS_clust(PData,orgdata,nclass,genes_data)
	        print('SC3 clustering complete')
		    ############################
		    
		    ari[p]=adjustedRandIndex(as.vector(as.matrix(orgclass$X)), as.vector(cl))   #### change cl$cluster for k-means
	print(ari[p])	    
}
print(ari)
