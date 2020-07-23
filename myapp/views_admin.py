from django.shortcuts import render, redirect

from sympy import false
from myapp.models import SysUser,VolunteerInfo,EventInfo,EmployeeInfo,OldpersonInfo

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