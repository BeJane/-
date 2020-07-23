import json
import time
from django.core import serializers

from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.shortcuts import render, redirect

from dwebsocket.decorators import accept_websocket,require_websocket


import numpy as np
from myapp.models import EventInfo


def events_info(request):
    name = request.session.get('name')

    list = np.arange(0,50)
    return render(request, 'events_info.html', {'name': name,'list':list})

@accept_websocket
def test_websocket(request):

    if not request.is_websocket():#判断是不是websocket连接
        try:#如果是普通的http方法
            message = request.GET['message']
            return HttpResponse(message)
        except:
            return render(request,'events_info.html')
    else:
        while 1:
            events = EventInfo.objects.all().order_by("-id")

            data = serializers.serialize("json", events)
            #data = list(data)
            request.websocket.send(data)#发送消息到客户端
            time.sleep(1)

@accept_websocket
def report(request):

    if not request.is_websocket():#判断是不是websocket连接
        try:#如果是普通的http方法
            message = request.GET['message']
            return HttpResponse(message)
        except:
            name = request.session.get('name')

            return render(request, 'event_report.html', {'name': name})
    else:

        while 1:

            data = {'type':[EventInfo.objects.filter(event_type=0).count(),
                         EventInfo.objects.filter(event_type=1).count(),
                            EventInfo.objects.filter(event_type=2).count(),
                            EventInfo.objects.filter(event_type=3).count(),
                            EventInfo.objects.filter(event_type=4).count()]}
            #data =serializers.serialize("json",data)
            request.websocket.send(json.dumps(data))#发送消息到客户端
            time.sleep(1)