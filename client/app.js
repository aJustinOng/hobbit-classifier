Dropzone.autoDiscover = false;
let hobbitArrayCounter = 0;
let match = [];

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Some Message",
        autoProcessQueue: false
    });

    dz.on("addedfile", function () {
        if (dz.files[1] != null) {
            dz.removeFile(dz.files[0]);
        }
    });

    dz.on("complete", function (file) {
        let imageData = file.dataURL;

        let url = "http://127.0.0.1:5000/classify_image";

        $.post(url, {
            image_data: file.dataURL
        }, function (data, status) {
            console.log(data);
            if (!data || data.length == 0) {
                $("#backBtn").hide();
                $("#nextBtn").hide();
                $("#resultHolder").hide();
                $("#divClassTable").hide();
                $("#error").show();
                return;
            }

            match = data.map(item => {
                let maxScore = Math.max(...item.class_probability);
                return { ...item, bestScore: maxScore };
            });

            // Show the first result by default
            hobbitArrayCounter = 0;
            updateDisplay();
        });
    });

    $("#submitBtn").on("click", function (e) {
        dz.processQueue();
    });

    $("#backBtn").on("click", function (e) {
        hobbitArrayCounter--;
        hobbitArrayCounter = (hobbitArrayCounter + match.length) % match.length; // cursed code bc js cannot do modulus -1 for some reason
        updateDisplay();
    });

    $("#nextBtn").on("click", function (e) {
        hobbitArrayCounter++;
        hobbitArrayCounter = hobbitArrayCounter % match.length;
        updateDisplay();
    });
}

function updateDisplay() {
    if (!match || match.length === 0) {
        return;
    }
    let currentMatch = match[hobbitArrayCounter];
    $("#backBtn").hide();
    $("#nextBtn").hide();
    $("#error").hide();
    $("#resultHolder").show();
    $("#divClassTable").show();
    if (match.length > 1) {
        $("#backBtn").show();
        $("#nextBtn").show();
    }

    $("#resultHolder").html($(`[data-player="${currentMatch.class}"]`).html());
    let classDictionary = currentMatch.class_dictionary;

    for (let personName in classDictionary) {
        let index = classDictionary[personName];
        let probabilityScore = currentMatch.class_probability[index];
        let elementName = "#score_" + personName;
        $(elementName).html(probabilityScore);
    }
}

$(document).ready(function () {
    console.log("ready!");
    $("#error").hide();
    $("#backBtn").hide();
    $("#nextBtn").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();

    init();
});