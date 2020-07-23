import json
import os

from django.http import StreamingHttpResponse
from django.shortcuts import render, redirect
from django.utils.timezone import now
import datetime


from sympy import false
from myapp.models import SysUser,VolunteerInfo,EmployeeInfo,OldpersonInfo

# Create your views here.
# 登录
def login(request):
    next_to = request.GET.get('next', false)
    if request.method == "POST":
        userName = request.POST.get("username")
        passWord = request.POST.get("password")
        if SysUser.objects.filter(username=userName,password=passWord).exists():
            request.session['username'] = userName
            user = SysUser.objects.get(username=userName)
            request.session['name'] = user.real_name
            request.session['uid'] = user.id
            return redirect('/main')
        else:
            return render(request,'login.html',{'message':'用户名/密码错误'})

    return render(request, 'login.html')
# 主页
def main(request):
    name = request.session.get('name')
    old_count = OldpersonInfo.objects.filter(remove=0).count()
    employee_count = EmployeeInfo.objects.filter(remove=0).count()
    volunteer_count = VolunteerInfo.objects.all().count()
    return render(request,'main.html',{'name':name,'old_count':old_count,
                                       'employee_count':employee_count,'volunteer_count':volunteer_count})
# 管理员信息维护
def personal(request):
    message = ''
    if request.method == "POST":
        username = request.POST.get("username")
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        mobile = request.POST.get('mobile')
        user = SysUser.objects.get(username=username)
        user.username = username
        user.real_name = name
        user.email = email
        user.phone = phone
        user.mobile = mobile

        user.save()
        message = '修改成功'
        request.session['name'] = user.real_name
        request.session['uid'] = user.id
    username = request.session.get('username')
    user = SysUser.objects.get(username=username)
    print(user.username)
    name = request.session.get('name')
    return render(request,'personal.html',{'user':user,'name':name,'message':message})
# 修改密码
def updatepassword(request):

    name = request.session.get('name')
    if request.method == "POST":
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        message = '两次输入密码不一致'
        if len(password1) < 6:
            message = '密码不能少于6位'
        elif password1 == password2:
            uid = request.session.get('uid')
            user = SysUser.objects.get(id=uid)
            user.password = password1
            user.save()
            message = '修改成功！'

        return render(request, 'update_password.html',{'name':name,'message':message})
    return render(request,'update_password.html',{'name':name})
# 新增老人信息
def add_old(request):

    name = request.session.get('name')
    if request.method == "POST":
        old_name = request.POST.get('name')
        sex = request.POST.get('sex')
        id_card = request.POST.get('id')
        birthday = request.POST.get('birth')
        health = request.POST.get('health')
        indate = request.POST.get('indate')
        room = request.POST.get('room')
        phone = request.POST.get('phone')

        name1 = request.POST.get('name1')
        relation1 = request.POST.get('relation1')
        phone1 = request.POST.get('phone1')
        wx1 = request.POST.get('wx1')
        name2 = request.POST.get('name2')
        relation2 = request.POST.get('relation2')
        phone2 = request.POST.get('phone2')
        wx2 = request.POST.get('wx2')

        bz = request.POST.get('bz')


        myFile = request.FILES.get("f")  # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            message = '请上传头像'

            return render(request, 'add_old.html', {'name': name, 'message': message})
        filename = '老人头像/'+old_name + room +'.jpg'
        path = os.path.join('static/', filename)
        destination = open(path, 'wb+')  # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():  # 分块写入文件
            destination.write(chunk)
        destination.close()

        uid = request.session.get('uid')

        OldpersonInfo.objects.create(username=old_name,gender=sex,phone=phone,profile_photo=filename,
                                     id_card=id_card,birthday=birthday,
                                     health_state=health, checkin_date=indate,
                                     room_number=room,firstguardian_name=name1,
                                     firstguardian_relationship=relation1,
                                     firstguardian_phone=phone1,firstguardian_wechat=wx1,
                                     secondguardian_name=name2,secondguardian_relationship=relation2,
                                     secondguardian_phone=phone2,secondguardian_wechat=wx2,
                                     description=bz, createby=uid,created=now(),remove=0)
        message = '创建成功'

        return render(request, 'add_old.html', {'name': name,'message':message})
    return render(request,'add_old.html',{'name':name})
# 老人信息管理
def olds_info(request):
    name = request.session.get('name')
    olds = OldpersonInfo.objects.filter(remove=0)
    return render(request,'olds_info.html',{'name':name,'olds':olds})
# 查看老人详细信息
def old_detail(request):
    id = request.GET['id']
    old = OldpersonInfo.objects.get(id=id)
    name = request.session.get('name')

    return render(request,'old_detail.html',{'name':name,'old':old})
# 删除老人信息
def old_delete(request):
    name = request.session.get('name')
    id = request.GET['id']
    old = OldpersonInfo.objects.get(id=id)
    old.remove = 1
    old.save()
    olds = OldpersonInfo.objects.filter(remove=0)
    return render(request,'olds_info.html',{'name':name,'olds':olds})
# 修改老人信息
def old_update(request):
    name = request.session.get('name')
    id = request.GET['id']

    old = OldpersonInfo.objects.get(id=id)
    if request.method == "POST":
        old.username = request.POST.get('name')
        old.gender = request.POST.get('sex')
        old.id_card = request.POST.get('id')
        old.birthday = request.POST.get('birth')
        old.health_state = request.POST.get('health')
        old.checkin_date = request.POST.get('indate')
        old.checkout_date = request.POST.get('outdate')
        old.room_number = request.POST.get('room')
        old.phone = request.POST.get('phone')

        old.firstguardian_name = request.POST.get('name1')
        old.firstguardian_relationship = request.POST.get('relation1')
        old.firstguardian_phone = request.POST.get('phone1')
        old.firstguardian_wechat = request.POST.get('wx1')
        old.secondguardian_name = request.POST.get('name2')
        old.secondguardian_relationship = request.POST.get('relation2')
        old.secondguardian_phone = request.POST.get('phone2')
        old.secondguardian_wechat = request.POST.get('wx2')

        old.description = request.POST.get('bz')
        old.updated = now()
        old.updateby = request.session.get('uid')

        myFile = request.FILES.get("f")  # 获取上传的文件，如果没有文件，则默认为None
        if  myFile:
            filename = '老人头像/'+old.username + old.room_number +'.jpg'
            path = os.path.join('static/', filename)
            destination = open(path, 'wb+')  # 打开特定的文件进行二进制的写操作
            for chunk in myFile.chunks():  # 分块写入文件
                destination.write(chunk)
            destination.close()
            old.profile_photo = filename
        old.save()
        message = '修改成功'
        old = OldpersonInfo.objects.get(id=id)
        return render(request,'old_update.html',{'name':name,'old':old,'message':message})

    return render(request,'old_update.html',{'name':name,'old':old})
# 老人信息统计分析
def old_analyze(request):
    name = request.session.get('name')
    agelist = [0,0,0,0]
    olds = OldpersonInfo.objects.filter(remove=0)
    ages = [datetime.date.today().year - old.birthday.year for old in olds]
    for age in ages:
        if age <60:
            agelist[0] = agelist[0] + 1
        elif age <= 70:
            agelist[1] = agelist[1] + 1
        elif age <= 80:
            agelist[2] = agelist[2] + 1
        else:
            agelist[3] = agelist[3] + 1

    genders = [old.gender for old in olds]
    genderlist = [0,0]

    for gender in genders:
        if gender == '男':
            genderlist[0] =genderlist[0] + 1
        else:
            genderlist[1] = genderlist[1] + 1

    healths = [old.health_state for old in olds]
    healthlist = [0,0]
    for health in healths:
        if health == '良好':
            healthlist[0] = healthlist[0] + 1
        else:
            healthlist[1] = healthlist[1] + 1
    return render(request,'old_analyze.html',{'name':name,'age':json.dumps(agelist),
                                              'gender':json.dumps(genderlist),'health':healthlist})



def video_stream():
    video_camera = VideoCamera()

    video_camera.start_record('static/video')
    while True:
        frame = video_camera.get_frame()

        if frame is not None:
            if frame is not None:
                global_frame = frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame
                       + b'\r\n\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n'
                       + global_frame + b'\r\n\r\n')
def camara(request):
    name = request.session.get('name')

    return render(request,'camara.html', {'name': name})

                  #content_type='multipart/x-mixed-replace; boundary=frame')
def video_viewer(request):


    return StreamingHttpResponse(video_stream(),  content_type='multipart/x-mixed-replace; boundary=frame')
    # 注意旧版的资料使用mimetype,现在已经改为content_type

