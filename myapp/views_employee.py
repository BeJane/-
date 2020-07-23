import json
import os

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.utils.timezone import now
import datetime
from sympy import false

from myapp.models import SysUser,EventInfo,EmployeeInfo,OldpersonInfo,VolunteerInfo

# Create your views here.

# 新增工作人员信息
def add_employee(request):
    name = request.session.get('name')
    if request.method == "POST":
        employee_name = request.POST.get('name')
        sex = request.POST.get('sex')
        id_card = request.POST.get('id')
        birthday = request.POST.get('birth')
        hiredate = request.POST.get('hiredate')

        phone = request.POST.get('phone')
        bz = request.POST.get('bz')

        myFile = request.FILES.get("f")  # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            message = '请上传头像'
            return render(request, 'add_old.html', {'name': name, 'message': message})
        filename = '工作人员头像/' + employee_name + phone + '.jpg'
        path = os.path.join('static/', filename)
        destination = open(path, 'wb+')  # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():  # 分块写入文件
            destination.write(chunk)
        destination.close()

        uid = request.session.get('uid')

        EmployeeInfo.objects.create(username=employee_name, gender=sex, phone=phone, profile_photo=filename,
                                    id_card=id_card, birthday=birthday, hire_date=hiredate,
                                    description=bz, createby=uid, created=now(), remove=0)
        message = '创建成功'

        return render(request, 'add_employee.html', {'name': name, 'message': message})
    return render(request, 'add_employee.html', {'name': name})


# 工作人员信息管理
def employees_info(request):
    name = request.session.get('name')
    employees = EmployeeInfo.objects.filter(remove=0)
    return render(request, 'employees_info.html', {'name': name, 'employees': employees})


# 查看工作人员详细信息
def employee_detail(request):
    id = request.GET['id']
    employee = EmployeeInfo.objects.get(id=id)
    name = request.session.get('name')

    return render(request, 'employee_detail.html', {'name': name, 'employee': employee})


# 删除工作人员信息
def employee_delete(request):
    name = request.session.get('name')
    id = request.GET['id']
    employee = EmployeeInfo.objects.get(id=id)
    employee.remove = 1
    employee.save()
    employees = EmployeeInfo.objects.filter(remove=0)
    return render(request, 'employees_info.html', {'name': name, 'employees': employees})


# 修改工作人员信息
def employee_update(request):
    name = request.session.get('name')
    id = request.GET['id']

    employee = EmployeeInfo.objects.get(id=id)
    if request.method == "POST":
        employee.username = request.POST.get('name')
        employee.gender = request.POST.get('sex')
        employee.id_card = request.POST.get('id')
        employee.birthday = request.POST.get('birth')

        employee.hiredate = request.POST.get('hiredate')
        employee.resign_date = request.POST.get('outdate')
        employee.phone = request.POST.get('phone')

        employee.description = request.POST.get('bz')
        employee.updated = now()
        employee.updateby = request.session.get('uid')

        myFile = request.FILES.get("f")  # 获取上传的文件，如果没有文件，则默认为None
        if myFile:
            filename = '工作人员头像/' + employee.username + employee.phone + '.jpg'
            path = os.path.join('static/', filename)
            destination = open(path, 'wb+')  # 打开特定的文件进行二进制的写操作
            for chunk in myFile.chunks():  # 分块写入文件
                destination.write(chunk)
            destination.close()
            employee.profile_photo = filename
        employee.save()
        message = '修改成功'
        employee = EmployeeInfo.objects.get(id=id)
        return render(request, 'employee_update.html', {'name': name, 'employee': employee, 'message': message})

    return render(request, 'employee_update.html', {'name': name, 'employee': employee})


# 工作人员信息统计分析
def employee_analyze(request):
    name = request.session.get('name')
    agelist = [0, 0, 0, 0]
    employees = EmployeeInfo.objects.filter(remove=0)
    ages = [datetime.date.today().year - employee.birthday.year for employee in employees]
    for age in ages:
        if age < 30:
            agelist[0] = agelist[0] + 1
        elif age <= 40:
            agelist[1] = agelist[1] + 1
        elif age <= 50:
            agelist[2] = agelist[2] + 1
        else:
            agelist[3] = agelist[3] + 1

    genders = [employee.gender for employee in employees]
    genderlist = [0, 0]

    for gender in genders:
        if gender == '男':
            genderlist[0] = genderlist[0] + 1
        else:
            genderlist[1] = genderlist[1] + 1

    return render(request, 'employee_analyze.html', {'name': name, 'age': json.dumps(agelist),
                                                     'gender': json.dumps(genderlist)})