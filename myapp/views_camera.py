import os
import subprocess
import time

from django.http import StreamingHttpResponse, HttpResponse
from django.shortcuts import render, redirect
from django.utils.timezone import now
from myapp.deskcamerautil import DeskCamera

from myapp.models import OldpersonInfo,EmployeeInfo,VolunteerInfo

from myapp.corridorcamerautil import CorridorCamera
from myapp.roomCamera import RoomCamera
from myapp.yardcamerautil import YardCamera

video_camera = []
video_camera.append(RoomCamera())
video_camera[0].__del__()#房间
video_camera.append(CorridorCamera())
video_camera[1].__del__()#走廊

video_camera.append(YardCamera())
video_camera[2].__del__()#院子

video_camera.append(DeskCamera())
video_camera[3].__del__()#桌子
state = -1
# Create your views here.

def openCamera(request):
    global state

    cid = int(request.GET['cid'])
    camera_id = state
    if camera_id != -1:
        video_camera[camera_id].stop_record()
        video_camera[camera_id].__del__()
        time.sleep(1)
    state = cid
    video_camera[cid].__init__()
    path = 'video/' + str(now()) + '.avi'
    video_camera[cid].start_record(path)

    return redirect('/camera')
#收集人脸数据
def collectFace(request):

    global state
    id = request.GET['id']
    type = request.GET['type']
    camera_id = state
    if camera_id != -1:
        video_camera[camera_id].stop_record()
        video_camera[camera_id].__del__()

    if type == '1': # 老人

        path = 'myapp/final/images/old_face/'+str(id)
        folder = os.path.exists(path)

        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        command = 'python myapp/final/collectingfaces.py --id %s --imagedir %s' % (str(id), 'myapp/final/images/old_face')
        p = subprocess.Popen(command, shell=True)

        time.sleep(10)
        state = -1
        old = OldpersonInfo.objects.get(id=id)
        old.imgset_dir = path
        old.save()

        return redirect('/olds_info')

    if type == '2': # 工作人员

        path = 'myapp/final/images/employee_face/'+str(id)
        folder = os.path.exists(path)

        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        command = 'python myapp/final/collectingfaces.py --id %s --imagedir %s' % (str(id), 'myapp/final/images/employee_face')
        p = subprocess.Popen(command, shell=True)
        time.sleep(10)
        state = -1
        entity = EmployeeInfo.objects.get(id=id)
        entity.imgset_dir = path
        entity.save()

        return redirect('/employees_info')

    if type == '3': # 义工

        path = 'myapp/final/images/volunteer_face/'+str(id)
        folder = os.path.exists(path)

        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


        command = 'python myapp/final/collectingfaces.py --id %s --imagedir %s' % (str(id), 'myapp/final/images/volunteer_face')
        p = subprocess.Popen(command, shell=True)

        time.sleep(10)
        state = -1
        entity = VolunteerInfo.objects.get(id=id)
        entity.imgset_dir = path
        entity.save()

        return redirect('/volunteers_info')
def close(request):

    global state
    camera_id = state
    video_camera[camera_id].stop_record()
    video_camera[camera_id].__del__()
    state = -1

    return redirect('/camera')
def video_stream(camera_id):

    while video_camera[camera_id] != None:
        frame = video_camera[camera_id].get_frame()
        if frame is not None:
            if frame is not None:
                global_frame = frame
                #print(frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame
                       + b'\r\n\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n'
                       + global_frame + b'\r\n\r\n')
def camera(request):
    name = request.session.get('name')

    return render(request,'camara.html', {'name': name})

                  #content_type='multipart/x-mixed-replace; boundary=frame')
def video_viewer(request):

    global state
    camera_id = state
    if camera_id == -1:
        return HttpResponse(None)
    else:
       return StreamingHttpResponse(video_stream(camera_id),  content_type='multipart/x-mixed-replace; boundary=frame')
        # 注意旧版的资料使用mimetype,现在已经改为content_type