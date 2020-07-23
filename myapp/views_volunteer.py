import json
import os

from django.shortcuts import render, redirect
from django.utils.timezone import now
import datetime

from myapp.models import VolunteerInfo

# Create your views here.

# 新增义工信息
def add_volunteer(request):
    name = request.session.get('name')
    if request.method == "POST":
        volunteer_name = request.POST.get('name')
        sex = request.POST.get('sex')
        id_card = request.POST.get('id')
        birthday = request.POST.get('birth')
        indate = request.POST.get('indate')

        phone = request.POST.get('phone')
        bz = request.POST.get('bz')

        myFile = request.FILES.get("f")  # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            message = '请上传头像'
            return render(request, 'add_volunteer.html', {'name': name, 'message': message})
        filename = '义工头像/' + volunteer_name + phone + '.jpg'
        path = os.path.join('static/', filename)
        destination = open(path, 'wb+')  # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():  # 分块写入文件
            destination.write(chunk)
        destination.close()

        uid = request.session.get('uid')

        VolunteerInfo.objects.create(name=volunteer_name, gender=sex, phone=phone, profile_photo=filename,
                                     id_card=id_card, birthday=birthday, checkin_date=indate,
                                     description=bz, createby=uid, created=now(), remove=0)
        message = '创建成功'

        return render(request, 'add_volunteer.html', {'name': name, 'message': message})
    return render(request, 'add_volunteer.html', {'name': name})


# 义工信息管理
def volunteers_info(request):
    name = request.session.get('name')
    volunteers = VolunteerInfo.objects.filter(remove=0)
    return render(request, 'volunteers_info.html', {'name': name, 'volunteers': volunteers})


# 查看义工详细信息
def volunteer_detail(request):
    id = request.GET['id']
    volunteer = VolunteerInfo.objects.get(id=id)
    name = request.session.get('name')

    return render(request, 'volunteer_detail.html', {'name': name, 'volunteer': volunteer})


# 删除义工信息
def volunteer_delete(request):
    name = request.session.get('name')
    id = request.GET['id']
    volunteer = VolunteerInfo.objects.get(id=id)
    volunteer.remove = 1
    volunteer.save()
    volunteers = VolunteerInfo.objects.filter(remove=0)
    return render(request, 'volunteers_info.html', {'name': name, 'volunteers': volunteers})


# 修改义工信息
def volunteer_update(request):
    name = request.session.get('name')
    id = request.GET['id']

    volunteer = VolunteerInfo.objects.get(id=id)
    if request.method == "POST":
        volunteer.name = request.POST.get('name')
        volunteer.gender = request.POST.get('sex')
        volunteer.id_card = request.POST.get('id')
        volunteer.birthday = request.POST.get('birth')

        volunteer.checkin_date = request.POST.get('indate')
        volunteer.checkout_date = request.POST.get('outdate')
        volunteer.phone = request.POST.get('phone')

        volunteer.description = request.POST.get('bz')
        volunteer.updated = now()
        volunteer.updateby = request.session.get('uid')

        myFile = request.FILES.get("f")  # 获取上传的文件，如果没有文件，则默认为None
        if myFile:
            filename = '义工头像/' + volunteer.name + volunteer.phone + '.jpg'
            path = os.path.join('static/', filename)
            destination = open(path, 'wb+')  # 打开特定的文件进行二进制的写操作
            for chunk in myFile.chunks():  # 分块写入文件
                destination.write(chunk)
            destination.close()
            volunteer.profile_photo = filename
        volunteer.save()
        message = '修改成功'
        volunteer = VolunteerInfo.objects.get(id=id)
        return render(request, 'volunteer_update.html', {'name': name, 'volunteer': volunteer, 'message': message})

    return render(request, 'volunteer_update.html', {'name': name, 'volunteer': volunteer})


# 义工信息统计分析
def volunteer_analyze(request):
    name = request.session.get('name')
    agelist = [0, 0, 0, 0]
    volunteers = VolunteerInfo.objects.filter(remove=0)
    ages = [datetime.date.today().year - volunteer.birthday.year for volunteer in volunteers]
    for age in ages:
        if age < 30:
            agelist[0] = agelist[0] + 1
        elif age <= 40:
            agelist[1] = agelist[1] + 1
        elif age <= 50:
            agelist[2] = agelist[2] + 1
        else:
            agelist[3] = agelist[3] + 1

    genders = [volunteer.gender for volunteer in volunteers]
    genderlist = [0, 0]

    for gender in genders:
        if gender == '男':
            genderlist[0] = genderlist[0] + 1
        else:
            genderlist[1] = genderlist[1] + 1

    return render(request, 'volunteer_analyze.html', {'name': name, 'age': json.dumps(agelist),
                                                      'gender': json.dumps(genderlist)})