"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from myapp import views_old,views_volunteer,views_employee,views_admin,views_event,views_camera

urlpatterns = [
    path('', views_admin.login),
    path('main', views_admin.main),
    path('personal', views_admin.personal),

    path('add_old', views_old.add_old),
    path('olds_info', views_old.olds_info),
    path('old_detail', views_old.old_detail),
    path('updatepassword', views_old.updatepassword), # 修改密码
    path('old_delete', views_old.old_delete), # 删除老人信息
    path('old_update', views_old.old_update),#修改老人信息
    path('old_analyze', views_old.old_analyze),#老人信息统计分析

    path('camera', views_camera.camera),
    path('video_viewer', views_camera.video_viewer),
    path('open',views_camera.openCamera),
    path('close',views_camera.close),
    path('facecollect',views_camera.collectFace),

    path('add_employee', views_employee.add_employee),
    path('add_volunteer', views_volunteer.add_volunteer),
    path('employees_info', views_employee.employees_info),
    path('volunteers_info', views_volunteer.volunteers_info),
    path('employee_detail', views_employee.employee_detail),
    path('volunteer_detail', views_volunteer.volunteer_detail),
    path('employee_update', views_employee.employee_update),
    path('volunteer_update', views_volunteer.volunteer_update),
    path('employee_analyze', views_employee.employee_analyze),
    path('volunteer_analyze', views_volunteer.volunteer_analyze),
    path('employee_delete', views_employee.employee_delete),
    path('volunteer_delete', views_volunteer.volunteer_delete),

    path('event_info',views_event.events_info),
    path('events',views_event.test_websocket),
    path('event_report',views_event.report)
]
