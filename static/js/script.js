function checkTime(i) {
    if (i < 10) {
        i = "0" + i;
    }
    return i;
}

function startTime() {
    var today = new Date();
    var h = today.getHours();
    var m = today.getMinutes();
    var s = today.getSeconds();
    // add a zero in front of numbers<10
    m = checkTime(m);
    s = checkTime(s);
    document.getElementById('time').innerHTML = h + ":" + m + ":" + s;
    t = setTimeout(function () {
        startTime()
    }, 500);
}

startTime();
get_last_detections();
function get_last_detections(){
    var feedback = $.ajax({
        type: "GET",
        url: "/get_last_detections",
        async: false
    }).complete(function(){
        setTimeout(function(){get_last_detections();}, 5000);
    }).responseText;

    $('.first-line').html(feedback);
}