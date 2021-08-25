import os
import igl


def points2ball(ball_out_path, v, ball_v, ball_f):
    fout = open(ball_out_path, 'w')
    for pid in range(v.shape[0]):
        p_ball_v = ball_v + v[pid:pid+1]
        for vid in range(ball_v.shape[0]):
            fout.write('v '+str(p_ball_v[vid][0])+' '+str(p_ball_v[vid]
                                                          [1])+' '+str(p_ball_v[vid][2])+'\n')  # 0 0 0\
    for pid in range(v.shape[0]):
        for fid in range(ball_f.shape[0]):
            fout.write('f '+str(ball_f[fid][0]+ball_v.shape[0]*pid+1)+' '+str(
                ball_f[fid][1]+ball_v.shape[0]*pid+1)+' '+str(ball_f[fid][2]+ball_v.shape[0]*pid+1)+'\n')
    fout.close()


# ball path
ball_path = './point2ball/s_ll.obj'
total_num = 70
ball_v, ball_f = igl.read_triangle_mesh(ball_path)

# input file
path = ".obj"
# out file
ball_out_path = '.obj'

v, _ = igl.read_triangle_mesh(path)
points2ball(ball_out_path, v, ball_v, ball_f)
