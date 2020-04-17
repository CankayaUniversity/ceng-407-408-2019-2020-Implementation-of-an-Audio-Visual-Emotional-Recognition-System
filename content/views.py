from django.shortcuts import render, get_object_or_404
from django.views.generic.edit import CreateView
from .models import Contents
from .forms import ContentsForm
from .visual_model import preprocessing, get_model, display
from .audio_model import process_audio
from .fusion_model import fusion


class ContentAdd(CreateView):
    model = Contents
    form_class = ContentsForm
    template_name = "content_add.html"
    success_url = "/list_contents/"


def content_list(request):
    contents = Contents.objects.order_by('id').reverse()
    return render(request, 'contents_list.html', {'contents': contents})


def content_detail(request, pk):
    post = get_object_or_404(Contents, pk=pk)
    return render(request, 'content_detail.html', {'content': post})


def analyze_content(request):
    return render(request, 'analyze_content.html', {})


def emotion_analyzes(request):
    labels = []
    data = []
    query = Contents.objects.all()
    for i in query:
        if i.content_emotion in labels:
            index = labels.index(i.content_emotion)
            data[index] += 1
        else:
            labels.append(i.content_emotion)
            data.append(1)
    return render(request, 'emotion_analyze.html', {
        'labels': labels,
        'data': data
    })


def language_analyzes(request):
    labels = []
    data = []
    query = Contents.objects.all()
    for i in query:
        if i.content_language in labels:
            index = labels.index(i.content_language)
            data[index] += 1
        else:
            labels.append(i.content_language)
            data.append(1)
    return render(request, 'language_analyze.html', {
        'labels': labels,
        'data': data
    })


def gender_analyzes(request):
    labels = []
    data = []
    query = Contents.objects.all()
    for i in query:
        if i.content_sex in labels:
            index = labels.index(i.content_sex)
            data[index] += 1
        else:
            labels.append(i.content_sex)
            data.append(1)
    return render(request, 'gender_analyze.html', {
        'labels': labels,
        'data': data
    })


def test_visual(request, pk):
    post = get_object_or_404(Contents, pk=pk)
    model = get_model()
    video = preprocessing('./' + post.content_upload.url)
    predict_video = model.predict(video, batch_size=2)
    result = display(predict_video)
    return render(request, 'test_visual.html', {
        'result': result,
        'content': post
    })


def test_audio(request, pk):
    post = get_object_or_404(Contents, pk=pk)
    # if post.content_type == "Audio":
    result = process_audio('./' + post.content_upload.url)
    """
    else:
        extract_audio(post.content_upload.url)
        result = process_audio("./media/contents/audio_extract.wav")
    """
    return render(request, 'test_audio.html', {
        'result': result,
        'content': post
    })


def test_fusion(request, pk):
    post = get_object_or_404(Contents, pk=pk)
    result = fusion(post)
    return render(request, 'test_fusion.html', {
        'result': result,
        'content': post
    })
