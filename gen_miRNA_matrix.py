# copyright: yueshi@usc.edu
import pandas as pd 
import hashlib
import os 
#from utils import logger
def file_as_bytes(file):
    with file:
        return file.read()

def extractMatrix(dirname):
	'''
	return a dataframe of the miRNA matrix, each row is the miRNA counts for a file_id

	'''
	count = 0

	miRNA_data = []
	for idname in os.listdir(dirname):
		# list all the ids 
		if idname.find("-") != -1:
			idpath = dirname +"/" + idname

			# all the files in each id directory
			for filename in os.listdir(idpath):
				# check the miRNA file
				if filename.find("-") != -1:

					filepath = idpath + "/" + filename
					df = pd.read_csv(filepath,sep="\t")
					# columns = ["miRNA_ID", "read_count"]
					if count ==0:
						# get the miRNA_IDs 
						miRNA_IDs = df.miRNA_ID.values.tolist()

					id_miRNA_read_counts = [idname] + df.read_count.values.tolist()
					miRNA_data.append(id_miRNA_read_counts)


					count +=1
					# print (df)
	columns = ["file_id"] + miRNA_IDs
	df = pd.DataFrame(miRNA_data, columns=columns)
	return df

def extractLabel(inputfile, inputfile2):
    df = pd.read_csv(inputfile, sep="\t")
    df2 = pd.read_csv(inputfile2, sep="\t")
    #
    # print (df[columns])
    df['label'] = df['cases.0.samples.0.sample_type']
    df.loc[df['cases.0.samples.0.sample_type'].str.contains("Normal"), 'label'] = "normal"
    tumors = df.loc[~df['cases.0.samples.0.sample_type'].str.contains("Normal"), 'cases.0.submitter_id']
    submitter_id = df2.iloc[:,0]
    disease_type = df2.iloc[:,19]
    primary_site = df2.iloc[:,108]
    # iterate over all rows to write new label values
    for row in tumors.iteritems():
            idx = submitter_id.index[submitter_id.str.contains(row[1])]
            df.loc[row[0],'label'] = primary_site[idx[0]] + ' ' + disease_type[idx[0]]
        
    #tumor_count = df.loc[df.label == 1].shape[0]
    #normal_count = df.loc[df.label == 0].shape[0]
    #logger.info("{} Normal samples, {} Tumor samples ".format(normal_count,tumor_count))
    columns = ['file_id','label']
    return df[columns]

if __name__ == '__main__':


	data_dir ="/Users/enriqueurdaneta/Desktop/Lab10_submit/"
	# Input directory and label file. The directory that holds the data. Modify this when use.
	dirname = data_dir + "live_miRNA"
	label_file = data_dir + "files_meta.tsv"
	label_file2 = data_dir + "cases_meta.tsv"
	
	#output file
	outputfile = data_dir + "miRNA_matrix.csv"

	# extract data
	matrix_df = extractMatrix(dirname)
	label_df = extractLabel(label_file, label_file2)

	#merge the two based on the file_id
	result = pd.merge(matrix_df, label_df, on='file_id', how="left")
	#print(result)

	#save data
	result.to_csv(outputfile, index=False)
	#print (labeldf)

 




