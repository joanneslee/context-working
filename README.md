# context-prediction-main

* deeplab_nokin_context is the script generates context labels based on the deeplab instrument masks without kinematics
* Data is in the same format as the DSA_Thread_Sampling repo

# File Structure
Tasks can be Needle_Passing, Knot_Tying, Suturing
Each task has its own data inputs and output folder, under which they are ordered by the usual ```<Task>_<Subject>_<Trial>```
* ```<task>```
    * deeplab_grasper_L_v3
    * deeplab_grasper_R_v3
    * deeplab_thread_v3
    * ctx_consensus
    * ctx_surgeon
    * vis_context_labels_v4
    * images
        * ```<Task>_<Subject>_<Trial>```
            * frame_0001.png

# Scripts

## deeplab_nokin_context.py

* JSONInterface_cogito: Helps to extract polygons, keypoints, and polylines from cogito Annotaiton JSON files
* JSONInterface_via: Helps to extract polygons, keypoints, and polylines from VGG Image Annotator (VIA) Annotaiton JSON files
* Iterator: loops through all images and generates context labels

## Notes:

install requirements2.txt

run `python deeplab_nokin_context_v2.py <Task name>

task name can be one of:
- Knot_Tying
- Needle_Passing
- Suturing

Iterator class:


