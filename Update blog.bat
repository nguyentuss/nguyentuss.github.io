 "D:\Work\obsidian-export\obsidian-export.exe" "D:\Work\Documents\Obsidian\obsidian-vault\AI" "D:\Work\nguyentuss.github.io\content\posts"
cd /d D:\Work\nguyentuss.github.io
hugo
git pull origin main
git add .
git commit -m "new"
git push -u origin main

echo Done!