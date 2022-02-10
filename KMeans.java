import java.util.Scanner;
import java.util.Arrays;


import java.io.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KMeans{
    
    public static int k = 0;

    public static class KMeansMapper extends Mapper<Object, Text, IntWritable, FloatWritable>{
        public String centroidsFile = "centroids.txt";
        public float[] centroids = new float[k];


        /* gets called by Mapper process 
           we're using it to initialize centroids array
           and read the centroids from the centroids file
           into the array
        */
        
        protected void setup(Context context) throws IOException {

            Scanner reader = new Scanner(new FileReader(centroidsFile));

            for (int i = 0; i < k; i++) {
                centroids[i] = reader.nextFloat();
            }
        }
        

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            //read value and store as float
            String val = value.toString();
            FloatWritable result = new FloatWritable();
            result.set(Float.valueOf(val).floatValue());

            float distance = 0;
            float center = 0;
            //a random and intentionally enormous value to initialize the inital min
            float min_dist = 100000.0f;
            int newbest = -1;
            int i = 0;
            
            for (i = 0; i < k; i++) {
                center = centroids[i];
                float num = result.get();

                //this basically just gets absolute value of the difference, the 1d euclidian distance
                distance = center > num ? center - num : num - center;
                
                if (distance < min_dist) {
                    min_dist = distance;
                    newbest = i;
                }
            }

            IntWritable best = new IntWritable(newbest);
            context.write(best, result);
        }
    }

    public static class KMeansReducer 
        extends Reducer<IntWritable, FloatWritable, IntWritable, FloatWritable> {
        
        public final static String centroidsFile = "centroids.txt";
        public float[] newcentroids = new float[k];

        public enum IS_CONVERGED{
            CONVERGED
        };

        public void reduce(IntWritable cluster, Iterable<FloatWritable> points, Context context) 
            throws IOException, InterruptedException {

            int num = 0;
            int index = cluster.get();
            float center = 0;
            
            for (FloatWritable point : points) {
                float x = point.get();
                center += x;
                num++;
            }

            center = center / num;
            newcentroids[index] = center;
            
            FloatWritable new_center = new FloatWritable(center);
            context.write(cluster, new_center);
        }

        protected void cleanup(Context context) throws IOException, InterruptedException {
            
            Scanner reader = new Scanner(new FileReader(centroidsFile));
            System.out.println(" ");

            int same = 1;

            for (int i = 0; i < k; i++){
                if (newcentroids[i] != reader.nextFloat()){
                    same = 0;
                }
            }
           

            if (same == 1){
                context.getCounter(IS_CONVERGED.CONVERGED).increment(1);
            }

            File file = new File("centroids.txt");

            file.createNewFile();
            FileWriter writer = new FileWriter(file);
            
        

            for (float val : newcentroids){
                writer.write(Float.toString(val));
                writer.write("\n");
            }

            writer.close();

        }
    }
    


    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();
        k = Integer.parseInt(args[0]);

        FileSystem fs   = FileSystem.get(conf);
        String datafile = "data.txt";
        

        BufferedReader bufferreader = new BufferedReader(new FileReader(datafile));

        File file = new File("centroids.txt");

        file.createNewFile();
        FileWriter writer = new FileWriter(file);

        for (int i = 0; i < k; i++) {
            writer.write(bufferreader.readLine());
            writer.write("\n");
        }

        writer.close();
        bufferreader.close();

        //Declare a job
        Job job; 

        long same = 0; 
        while (same != 1){

            if(fs.exists(new Path(args[2]))){
                fs.delete(new Path(args[2]),true);
            }

            job = Job.getInstance(conf, "kmeans");
            job.setJarByClass(KMeans.class);
            job.setMapperClass(KMeansMapper.class);
            //job.setCombinerClass(KMeansReducer.class);
            job.setReducerClass(KMeansReducer.class);
            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(FloatWritable.class);
            FileInputFormat.addInputPath(job, new Path(args[1]));
            FileOutputFormat.setOutputPath(job, new Path(args[2]));
            job.waitForCompletion(true);
            same = job.getCounters().findCounter(KMeansReducer.IS_CONVERGED.CONVERGED).getValue();
            
        }
    
    }
}