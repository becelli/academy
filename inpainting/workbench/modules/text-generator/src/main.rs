use markov::Chain;

fn main() {
    let mut chain = Chain::new();

    // read from assets/markov_seed.txt
    let text = std::fs::read_to_string("markov_seed.txt").unwrap();
    chain.feed_str(&text);
    // generate 10 sentences
    let generated = chain.generate();
    println!("{}", generated.join(" "));
}
