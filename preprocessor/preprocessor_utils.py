import numpy as np
import cv2

def normalize(label, landmark, resolution, face_img, face_mask, origin):
    label = label * resolution
    
    center = (np.mean(landmark[:, 0]), np.mean(landmark[:, 1]))
    angle = np.arctan((landmark[0, 1]-landmark[16, 1])/(landmark[0, 0]-landmark[16, 0]+0.01))*180/np.pi
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)


    face_img = cv2.warpAffine(face_img, rot_mat, (face_img.shape[1], face_img.shape[0]))
    face_mask = cv2.warpAffine(face_mask, rot_mat, (face_mask.shape[1], face_mask.shape[0]))
    crop_img = cv2.warpAffine(origin, rot_mat, (origin.shape[1], origin.shape[0]))


    scale = np.sqrt(np.sum((label[36, :]-label[45, :])**2))/np.sqrt(np.sum((landmark[36, :]-landmark[45, :])**2))
    lab_center = (np.mean(label[:, 0]), np.mean(label[:, 1]))
    
    
    ali_mat = np.array([[0, 0, lab_center[0]-center[0]], [0, 0, lab_center[1]-center[1]]]).astype(np.float32)
    trs_mat = np.array([[scale, 0, center[0]*(1-scale)], [0, scale, center[1]*(1-scale)]]).astype(np.float32)
    
    
    face_img = cv2.warpAffine(face_img, ali_mat + trs_mat, (resolution, resolution))
    face_mask = cv2.warpAffine(face_mask, ali_mat + trs_mat, (resolution, resolution))
    crop_img = cv2.warpAffine(crop_img, ali_mat + trs_mat, (resolution, resolution))
    norm_mats = (rot_mat, ali_mat, trs_mat)


    return face_img, face_mask, crop_img, norm_mats
    

def fill_mask_mouth(face_mask):
    face_mask_copy = face_mask.copy() 
    mask = np.zeros((face_mask.shape[0]+2,face_mask.shape[1]+2)).astype(np.uint8)
    cv2.floodFill(face_mask, mask, (0, 0), (1,1,1), cv2.FLOODFILL_MASK_ONLY) 
    mouth_mask = 1 - face_mask
    # face_mask_copy: face with no mouth
    # face_mask: everywhere with no mouth
    # mouth_mask: only nouth
    face_mask = face_mask_copy + mouth_mask
    return face_mask
