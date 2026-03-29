from django.shortcuts import render, redirect
from .models import modeldata
from .forms import modeldataForm
from django.contrib import messages
from django.db import IntegrityError

def adminhome(request):
    return render(request, 'admins/adminhome.html')

def register(request):
    if request.method == 'POST':
        form = modeldataForm(request.POST)
        if form.is_valid():
            try:
                form.save()
                form = modeldataForm()
                messages.success(request, 'Registered Successfully! Please wait for admin activation.')
                return render(request, 'register.html', {'form': form, 'message': 'Registered Successfully'})
            except IntegrityError:
                messages.error(request, 'Username already exists. Please choose another one.')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = modeldataForm()
    return render(request, 'register.html', {'form': form})


def view(request):
    modeldataa = modeldata.objects.all()
    return render(request, 'admins/view.html', {'modeldataA': modeldataa})

def activate(request, id):
    if request.method == 'GET':
        if id is not None:
            status = 'Activated'
            print("PID = ", id, status)
            modeldata.objects.filter(id=id).update(status=status)
            return redirect('view')
        
def block(request, id):
    if request.method == 'GET':
        if id is not None:
            status = 'waiting'
            print("PID = ", id, status)
            modeldata.objects.filter(id=id).update(status=status)
            return redirect('view')

def adminlogin(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()
        print(f"--- Admin Login Diagnostic ---")
        print(f"Attempted Admin: [{username}]")
        
        if username == 'admin' and password == 'admin':
            print("SUCCESS: Admin credentials verified.")
            return redirect('adminhome')
        else:
            print(f"FAILED: Admin mismatch. Expected 'admin'/'admin', got '{username}'/'{password}'")
            if username.lower() == 'admin':
                messages.error(request, 'This is the Admin Portal. Please use the User Access page to login as a regular user, or check admin credentials.')
            else:
                messages.error(request, 'Invalid Credentials.')
    return render(request, 'adminlogin.html')

def adminbase(request):
    return render(request, 'admins/adminbase.html')

def delete(request, id):
    if request.method == 'GET':
        if id is not None:
            modeldata.objects.filter(id=id).delete()
            return redirect('view')