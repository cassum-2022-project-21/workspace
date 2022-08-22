for d in `ls -d */`; do
    python -c "import json; f=open('$d/args.json', 'w'); Namespace = lambda **kwargs: json.dump(kwargs, f, indent=2); $(head -n 2 $d/output.txt | tail -n 1)"
done