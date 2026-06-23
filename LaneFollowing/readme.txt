* YOU MUST INSTALL PYTORCH BEFORE START THIS.!!

* IF NOT : 
   go to tis page to Install DL setting 
   
   https://github.com/katebrighteyes/jetbot_new/tree/main/StartDL
   

1. data_collection

1) Take road Picture

python3 lane_data_collection.py

# click the road center and push 's' button to save labeled image
(collect picture more than 70)

# Ctl-C to finish it

----------------------------------------
2. train_road foloww_model

python3 train_lf_model.py

----------------------------------------

* YOU MUST set power mod 5W !!

3. lf_live_demo

y = -((0.5 - xy[1]) / 2.0)

python3 lf_live_demo.py


https://drive.google.com/file/d/1bAB3PK-m_4BmhU_clNuZYhD_cH_RQm2O/view?usp=sharing

