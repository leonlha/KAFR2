from comet_ml import Experiment
from ultralytics import YOLO


# import wandb
# from wandb.integration.yolov8 import add_callbacks as add_wandb_callbacks


# wandb.login()

## if you want to track experiment with Comet Input the Api key or else comment the experiment and api key

if __name__=='__main__':
    
    COMET_API_KEY = ''
    experiment = Experiment(
      api_key=COMET_API_KEY,
      project_name="crural_project", workspace= "shekhar07",
      auto_param_logging=True, auto_metric_logging=True)

      
    params = {'epochs': 500, 'patience': 200, 'batch': 4, 'imgsz': 640, 'save': 'true', 'save_period': -1, 'cache': 'false', 'device': 'null',
               'project': 'null', 'name': 'yolov8mwithaug_crural1.5', 'exist_ok': 'false', 'pretrained': 'false', 'optimizer': 'SGD', 'verbose': 'true',
              'seed': 0, 'deterministic': 'true', 'single_cls': 'false', 'image_weights': 'false', 'cos_lr': 'true', 'close_mosaic': 10,
              'resume': 'false', 'min_memory': 'false', 'overlap_mask': 'true', 'mask_ratio': 4, 'dropout': 0.0, 'val': 'true', 'split': 'val',
              'save_json': 'false', 'lr0' : 0.01,'lrf' : 0.01,'save_hybrid': 'false', 'conf': 'null', 'iou': 0.7, 'max_det': 300, 'plots': 'true',
              'save_txt': 'true', 'save_conf': 'false','vid_stride': 1, 'line_thickness': 3, 'visualize': 'false', 'augment': 'false',
              'agnostic_nms': 'false', 'classes': 'null', 'retina_masks': 'false', 'boxes': 'true','optimize': 'false', 'dynamic': 'false', 'simplify': 'false', 'opset': 'null',
              'nms': 'false', 'label_smoothing': 0.0,'nbs': 64, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 
              'degrees': 0.0, 'translate': 0.1, 'scale': 0.5,'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.2, 'mosaic': 0.0, 'mixup': 0.0}

    experiment.log_parameters(params)
    experiment.set_name("yolov8mwithaug_crural1.5")
    
    
    
    # Load a model

    model = YOLO(r"C:\Users\s220082\Deep_Learning_Projects\crural_repair\yolov8m-seg.pt")  # load a pretrained model 


    #Use the model
    results = model.train(data=r"C:\Users\s220082\Deep_Learning_Projects\crural_repair\data.yaml", epochs= 500
                          ,patience= 200,batch= 4,imgsz= 640,save= True,
                          save_period= -1,cache= False,
                          name= 'yolov8mwithaug_crural1.5',exist_ok= False,pretrained= False,optimizer= 'SGD',verbose= True,seed= 0,deterministic= True,
                          single_cls= False,cos_lr=False,close_mosaic= 10,overlap_mask= True,mask_ratio= 4,
                          dropout= 0.0,val= True,split= 'val',iou= 0.7,max_det= 300,half= False,dnn= False,plots= True,
                          show= False,save_txt= True,save_conf= True,save_crop= False,vid_stride= 1,
                          line_width= 3,visualize= False,augment= False,agnostic_nms= False,retina_masks= False,
                          boxes= True,format= 'torchscript',optimize= False,
                          nms= False,lr0= 0.01,lrf= 0.01,momentum= 0.937,weight_decay= 0.0005,warmup_epochs= 3.0,warmup_momentum= 0.8,
                          warmup_bias_lr= 0.1,box= 7.5,cls= 0.5,dfl= 1.5,label_smoothing= 0.0,
                          nbs= 64,hsv_h= 0.015,hsv_s= 0.7,hsv_v= 0.4,degrees= 0.0,translate= 0.1,scale= 0.5,
                          shear= 0.0,perspective= 0.0,flipud= 0.0,fliplr= 0.2,mosaic= 0.0,mixup= 0.0,copy_paste= 0.0)  # train the model

    # model = YOLO(r"C:\Users\s220082\Deep_Learning_Projects\crural_repair\runs\segment\yolov8mwithaug_crural1.4\weights\best.pt")  # load a pretrained model 
    
    results = model.val(split= 'test')

    experiment.end()

