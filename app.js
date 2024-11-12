document.getElementById('submit-btn').addEventListener('click', function() {
    const question = document.getElementById('question').value;

    if (!question.trim()) {
        alert("Veuillez entrer une question");
        return;
    }

    // Envoi de la question à l'API et réception de la réponse
    fetch('https://mon-api-ai-biblique.com/reponse', {  // Remplace par ton backend réel
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('response-container').innerHTML = data.reponse;
    })
    .catch(error => {
        console.error('Erreur:', error);
        document.getElementById('response-container').innerHTML = "Une erreur est survenue.";
    });
});
