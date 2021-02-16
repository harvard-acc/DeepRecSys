# DeepRecSys: A System for Optimizing End-To-End At-scale Neural Recommendation Inference
DeepRecSys provides an end-to-end infrastructure to study and optimize at-scale neural recommendation inference.
The infrastructure is configurable across three main dimensions that represent different recommendation use cases: the load generator (query arrival patterns and size distributions), neural recommendation models, and underlying hardware platforms.

## Neural recommendation models
This repository supports 8-industry representative neural recommendation models based on open-source publications from various Internet services in Caffe2:

1. Deep Learning Recommendation Models (DLRM-RMC1, DLRM-RMC2, DLRM-RMC3); [link](https://arxiv.org/abs/1906.03109)
2. Neural Collaborative Filtering (NCF); [link](https://arxiv.org/pdf/1708.05031.pdf)
3. Wide and Deep (WnD); [link](https://arxiv.org/pdf/1606.07792.pdf)
4. Multi-task Wide and Deep (MT-WnD); [link](https://daiwk.github.io/assets/youtube-multitask.pdf)
5. Deep Interest Network (DIN); [link](https://arxiv.org/pdf/1706.06978.pdf)
6. Deep Interest Evolution Network (DIEN); [link](https://arxiv.org/pdf/1809.03672.pdf)

## Getting started
To get you started quickly, we have provided a number of examples scripts to run synthetic models, characterize hardware platforms, model at-scale inference, and optimizing scheduling decisions. 

The code is structured such that it enables maximum flexibility for future extensions. 
1. The top-level is found in ```DeepRecSys.py```. This co-ordinates the models, load generator, scheduler, and hardware backends.
2. Models can be found in the ```models``` directory.
3. The load generator is in ```loadGenerator.py```
4. The scheduler is in ```scheduler.py```
5. The CPU and accelerator inference engines are found in ```inferenceEngine.py``` and ```accelInferenceEngine.py``` respectively. 

You can build the necessary python packages, using conda or docker environments, based on ```build/pip_requirements.txt```. 

### Characterizing performance of neural recommendation models
To run the individual models you may use the ```models/run.sh``` script directly.
In addition, we have provided two experiments to characterize the neural recommendation models.
First, ```experiments/operator_breakdown/sweep_p.py``` generates the operator breakdown of each model running on a CPU and GPU.
Second, ```experiments/speedup/sweep_rt.py``` generates the speedup of accelerator platforms (like GPUs) over CPUs for the provided models.

### Modeling at-scale inference
To model at-scale inference we provide a sample script, ```run_DeepRecInfra.sh```.
This runs the end-to-end system using ```DeepRecSys.py``` with an example model, query input arrival and size distributions for the load generator, on CPU-only as well as CPU and accelerator-enabled nodes.
In order to run with the accelerator-enabled nodes, please first run ```accelerator/<accelName>/generate_data.py``` followed by ```accelerator/predict_execution.py```. 
The ```run_DeepRecInfra.sh``` script outputs the measured tail-latency of queries.
Note this example does not include the recommendation query scheduler that optimizes inference QPS under a strict tail-latency target.

### Optimizing inference QPS
To optimize inference QPS under strict tail-latency targets with the scheduler, we provide an example in ```run_DeepRecSys.sh```.
Following the same input characteristics as ```run_DeepRecInfra.sh``` this example incorporates the query scheduler across CPU cores (balancing data-level and thread-level parallelism) and the simulated accelerator nodes (offloading queries to specialized hardware).

## Link to paper
To read the paper please visit this [link](http://vlsiarch.eecs.harvard.edu/wp-content/uploads/2020/05/DeepRecSys_Gupta_ISCA2020.pdf)

## Citation
If you use `DeepRecSys`, please cite us:

```
   @conference{Gupta2020b,
   title = {DeepRecSys: A System for Optimizing End-To-End At-scale Neural Recommendation Inference},
   author = {Udit Gupta, Samuel Hsia, Vikram Saraph, Xiaodong Wang, Brandon Reagen, Gu-Yeon Wei, Hsien-Hsin S. Lee, David Brooks, Carole-Jean Wu
   },
   url = {http://vlsiarch.eecs.harvard.edu/wp-content/uploads/2020/05/DeepRecSys_Gupta_ISCA2020.pdf},
   year = {2020},
   date = {2020-06-01},
   publisher = {The 47th IEEE/ACM International Symposium on Computer Architecture (ISCA 2020)},
   abstract = {Neural personalized recommendation is the corner-stone of a wide collection of cloud services and products, constituting significant compute demand of the cloud infrastructure. Thus, improving the execution efficiency of neural recommendation directly translates into infrastructure capacity saving. In this paper, we devise a novel end-to-end modeling infrastructure, DeepRecInfra, that adopts an algorithm and system co-design methodology to custom-design systems for recommendation use cases. Leveraging the insights from the recommendation characterization, a new dynamic scheduler, DeepRecSched, is proposed to maximize latency-bounded throughput by taking into account characteristics of inference query size and arrival patterns, recommendation model architectures, and underlying hardware systems. By doing so, system throughput is doubled across the eight industry-representative recommendation models. Finally, design, deployment, and evaluation in at-scale production datacenter shows over 30% latency reduction across a wide variety of recommendation models running on hundreds of machines.},
   keywords = {},
   pubstate = {published},
   tppubtype = {conference}
   }
 ```

## Contact Us
For any further questions please contact <ugupta@g.harvard.edu>, <shsia@g.harvard.edu>, or <carolejeanwu@fb.com>
