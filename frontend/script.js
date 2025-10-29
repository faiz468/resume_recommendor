
// Attach click handlers to cards
document.addEventListener("DOMContentLoaded", function(){
  const cards = document.querySelectorAll(".card");
  const modal = document.getElementById("modal");
  const modalBody = document.getElementById("modalBody");
  const closeBtn = document.getElementById("closeModal");
  const modalApply = document.getElementById("modalApply");

  cards.forEach(c => {
    c.addEventListener("click", function(e){
      // Only open modal when clicking outside the Apply button
      if(e.target.closest(".apply-btn")) return;
      const desc = c.getAttribute("data-desc");
      const url = c.getAttribute("data-url");
      let parsedDesc = desc;
      try{ parsedDesc = JSON.parse(desc); }catch(e){}
      let parsedUrl = url;
      try{ parsedUrl = JSON.parse(url); }catch(e){}

      modalBody.innerHTML = "";
      // Simple formatting: keep paragraphs, preserve whitespace
      const node = document.createElement("div");
      node.innerText = parsedDesc || "No description available.";
      modalBody.appendChild(node);

      modalApply.href = parsedUrl || "#";
      modal.setAttribute("aria-hidden", "false");
    });
  });

  closeBtn.addEventListener("click", function(){ modal.setAttribute("aria-hidden", "true"); });
  modal.addEventListener("click", function(e){
    if(e.target === modal) modal.setAttribute("aria-hidden", "true");
  });
});
