namespace Assets.Scripts.IAJ.Unity.Movement.DynamicMovement
{
    public class DynamicSeek : DynamicMovement
    {
        public override string Name
        {
            get { return "Seek"; }
        }

        public DynamicSeek()
        {
            this.Output = new MovementOutput();
        }

        public override MovementOutput GetMovement()
        {

            // Get the direction to the targfet
            this.Output.linear = this.Target.Position - this.Character.Position;

            // Give full acceleration along this direction
            this.Output.linear.Normalize();
            this.Output.linear *= this.Character.MaxAcceleration;

            // We could add a stopping condition here but it would look janky

            /*if (this.Output.linear.sqrMagnitude < 2)

               this.Character.velocity = UnityEngine.Vector3.zero;
           }*/
            this.Output.angular = 0;

            return this.Output;
        }
    }
}
