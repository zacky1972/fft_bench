inputs =
  [1024]
  |> Enum.map(fn size ->
    {
      "size #{size}: IPS #{div(105_000_000, size) |> Number.Delimit.number_to_delimited()}",
      Nx.random_uniform({size}, 0, 65536, type: {:s, 32})
    }
  end)
  |> Map.new()

Benchee.run(
  %{
    "FFT" => fn input -> Nx.fft(input) end
  },
  inputs: inputs
)
